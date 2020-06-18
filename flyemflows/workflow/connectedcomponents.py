import os
import sys
import copy
import pickle
import logging

import h5py
import numpy as np
import pandas as pd
import skimage.measure as skm
from dask import delayed
from dask.bag import zip as bag_zip
import dask.dataframe as ddf

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, connected_components_nonconsecutive, apply_mask_for_labels, block_stats_for_volume, BLOCK_STATS_DTYPES, round_box, SparseBlockMask
from neuclease.dvid import fetch_instance_info, fetch_maxlabel, post_maxlabel, post_nextlabel, fetch_roi

from dvidutils import LabelMapper

from ..util import replace_default_entries
from ..brick import Brick, BrickWall, extract_halos, PB_COLS, clip_to_logical
from ..volumes import VolumeService, VolumeServiceWriter, SegmentationVolumeSchema, DvidVolumeService, ScaledVolumeService
from .util.config_helpers import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)

class ConnectedComponents(Workflow):
    """
    Analyze a segmentation volume in which some objects may
    consist of multiple disconnected pieces, and relabel
    those pieces to give them unique IDs.
    """
    ConnectedComponentsOptionsSchema = \
    {
        "type": "object",
        "description": "Settings specific to the ConnectedComponents workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "subset-labels": {
                "description": "If provided, only the listed labels will be analyzed for connected components analysis.\n"
                               "Other labels will be left untouched in the results.\n",
                **BodyListSchema
            },
            "skip-sparse-fetch": {
                "description": "If True, do not attempt to fetch the sparsevol-coarse for the given subset-labels.\n"
                               "Just fetch the entire bounding-box.\n",
                "type": "boolean",
                "default": False
            },
            "roi": {
                "description": "Limit analysis to bricks that intersect the given DVID ROI.\n",
                "type": "object",
                "default": {},
                "properties": {
                    "server": {
                        "description": "dvid server for the ROI. If not provided, the input server will be used (if possible).",
                        "type": "string",
                        "default": ""
                    },
                    "uuid": {
                        "description": "dvid UUID for the ROI.  If not provided, the input UUID will be used (if possible).",
                        "type": "string",
                        "default": ""
                    },
                    "name": {
                        "description": "name of the ROI",
                        "type": "string",
                        "default": ""
                    }
                }
            },
            "halo": {
                "description": "How much overlapping context between bricks in the grid (in voxels)\n",
                "type": "integer",
                "minValue": 1,
                "default": 1
            },
            "orig-max-label": {
                "description": "Relabeled objects will be given new labels starting after this value.\n"
                               "If provided, this value should be chosen to be high enough not to conflict\n"
                               "with any other objects in the volume (even outside the bounding box you're working with).\n"
                               "The default (0) has a special meaning, which results in an\n"
                               "automatically chosen starting value for new labels.\n"
                               "In the automatic case, special support is implemented for DVID volumes,\n"
                               "to ensure that the necessary labels are reserved in advance.",
                "type": "integer",
                "default": 0
            },
            "compute-block-statistics": {
                "description": "Whether or not to compute block statistics on the output blocks.\n",
                "type": "boolean",
                "default": True
            },
            "block-statistics-file": {
                "description": "Where to store block statistics for the output segmentation\n"
                               "Supported formats: .csv and .h5\n",
                "type": "string",
                "default": "block-statistics.h5"
            },
            "log-relabeled-objects": {
                "description": "Write relabeled object mapping to CSV.\n",
                "type": "boolean",
                "default": False
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,
        "output": SegmentationVolumeSchema,
        "connectedcomponents": ConnectedComponentsOptionsSchema
    })


    @classmethod
    def schema(cls):
        return ConnectedComponents.Schema

    def execute(self):
        """
        Computes connected components across an entire volume,
        possibly considering only a subset of all labels,
        and writes the result to another volume.
        
        (Even if only some labels were analyzed for CC, all labels are written to the output.)
        
        Objects that were not "disconnected" in the first place are not modified.
        Their voxels retain their original values and they are simply copied to
        the output.  Only objects which consist of multiple disconnected pieces
        are given new label values.
        
        Procedure Overview:

        1. Accepts any segmentation volume as input, divided into
           'bricks' as usual, but with a halo of at least 1 voxel.
        
        2. Computes connected components (CC) on each brick independently.
           Note: Before computing the CC, the "subset-labels" setting is used to
           mask out everything except the desired objects.
           Note that the resulting CC labels for each brick are not unique -- each
           brick has values of 1..cc_max.
           Also, the original label from which each CC label was created is stored in
           the "overlap mapping" (dataframe for each brick with columns: ['cc', 'orig']).
           The CC volume, overlap mapping, cc_max, and raw data max
           for each brick are cached (persisted).
        
        3. The CC values are made unique across all bricks by adding an offset to each brick.
           The offset is determined by multiplying the brick's ID number (scan-order index)
           with the maximum CC value found in all the brick CC volumes.
           Note: This means that the CC values will be unique, but not consecutive across all bricks.
           The overlap mapping (cc->orig) is also updated accordingly.
        
        4. Next, we need to determine how to unify objects which span across brick boundaries.
           First, the halos from each brick are extracted and matched with the halos from 
           their neighboring bricks. This is achieved via a dask DataFrame merge.
           Then these halo pairs are aligned so that the CC label in the "left" halo can
           be matched with the CC label in the "right" halo.
           These pairings "link" the CC objects in one brick to another brick,
           and are referred to as "pairwise links".
        
        5. These links form the edges of a graph, and the connected components *on the graph*
           determine which CC labels should be unified in the output.
           Note that here, we are referring to a graph-CC, not to be confused with the
           connected components operations we ran earlier, on each brick's volume data.
           This graph-CC operation yields a "final mapping" from brick-CC labels to unified
           label values.
           Note: Since we are not interested in relabeling objects that weren't
           discontiguous in the original volume, we drop rows of the mapping for objects whose
           'orig' value only appears once in the final mapping.  This reduces the size of the
           mapping, which must be sent to the workers.  In fact, before we even run the graph-CC
           operation, we drop CC values if their original label doesn't appear more than once in
           the overlap mapping from above. This reduces the size of the graph-CC problem.
        
        6. The final mapping is distrbuted to all workers in pieces, via pandas DataFrame merge.
        
        7. The final mapping is applied to the CC bricks, and written to the output.
           If the "subset-labels" option was used, the CC brick may consist of zeros for those
           voxels that did not contain a label of interest (see step 1, above).  But the output
           is guaranteed to contain all of the unsplit original objects, so we use the original
           volume to supply the remaining data before it is written to the output.
           
        """
        # TODO:
        #
        #  - Refactor this super-long function.
        #
        #  - Maybe parallelize some of those dataframe operations,
        #    rather than collecting it all on the driver right away?
        #
        #  - Don't bother including LB_COLS in dataframes? (see BrickWall.bricks_as_ddf)
        #
        #  - Unpersist unneeded bags? (maybe I need them all, though...)
        #
        #  - Refactor this so that the bulk of the code is re-usable for workflows in which 
        #    the segmentation did not come from a common source.
        #    (But see notes about necessary changes below.)
        #
        #  - Right now, voxels OUTSIDE the subset-labels are copied verbatim into the output.
        #    If the output segmentation is different from the input segmentation,
        #    it might be nice to fetch the output bricks and then paste the subset mask on top of it,
        #    rather than copying from the input mask.
        #
        #  - For DVID volumes, this workflow only writes supervoxels, not labels.
        #    To write labels, one would need to first split supervoxels (if necessary) via this workflow,
        #    and then partition complete labels according to groups of supervoxels.
        #    That might be best performed via the FindAdjacencies workflow, anyhow.

        self.init_services()

        input_service = self.input_service
        output_service = self.output_service
        options = self.config["connectedcomponents"]

        is_supervoxels = False
        if isinstance(input_service.base_service, DvidVolumeService):
            is_supervoxels = input_service.base_service.supervoxels

        # Load body list and eliminate duplicates
        subset_labels = load_body_list(options["subset-labels"], is_supervoxels)
        subset_labels = set(subset_labels)
        
        sparse_fetch = not options["skip-sparse-fetch"]
        input_wall = self.init_brickwall(input_service, sparse_fetch and subset_labels, options["roi"])
        
        def brick_cc(brick):
            orig_vol = brick.volume
            brick.compress()

            # Track the original max so we know what the first
            # available label is when we write the final results.
            orig_max = orig_vol.max()
            
            if subset_labels:
                orig_vol = apply_mask_for_labels(orig_vol, subset_labels)
            
            # Fast path for all-zero bricks
            if not orig_vol.any():
                cc_vol = orig_vol
                cc_overlaps = pd.DataFrame({'orig': [], 'cc': []}, dtype=np.uint64)
                cc_max = np.uint64(0)
            else:
                cc_vol = skm.label(orig_vol, background=0, connectivity=1)
                assert cc_vol.dtype == np.int64
                cc_vol = cc_vol.view(np.uint64)
                
                # Leave 0-pixels alone.
                cc_vol[orig_vol == 0] = np.uint64(0)
                
                # Keep track of which original values each cc corresponds to.
                cc_overlaps = pd.DataFrame({'orig': orig_vol.reshape(-1), 'cc': cc_vol.reshape(-1)})
                cc_overlaps.query('orig != 0 and cc != 0', inplace=True)
                cc_overlaps = cc_overlaps.drop_duplicates()
                assert (cc_overlaps.dtypes == np.uint64).all()
    
                if len(cc_overlaps) > 0:
                    cc_max = cc_overlaps['cc'].max()
                else:
                    cc_max = np.uint64(0)
            
            cc_brick = Brick( brick.logical_box,
                              brick.physical_box,
                              cc_vol,
                              location_id=brick.location_id,
                              compression=brick.compression )

            return cc_brick, cc_overlaps, cc_max, orig_max

        cc_results = input_wall.bricks.map(brick_cc)
        cc_results = cc_results.persist()
            
        cc_bricks, cc_overlaps, cc_maxes, orig_maxes = cc_results.unzip(4)

        with Timer("Computing blockwise CC", logger):
            max_brick_cc = cc_maxes.max().compute()
        
        with Timer("Saving brick maxes", logger):
            def corner_and_maxes(cc_brick, _cc_overlaps, cc_max, orig_max):
                return (*cc_brick.logical_box[0], cc_max, orig_max)
            brick_maxes = cc_results.starmap(corner_and_maxes).compute()
            brick_maxes_df = pd.DataFrame(brick_maxes, columns=['z', 'y', 'x', 'cc_max', 'orig_max'])
            brick_maxes_df.to_csv('brick-maxes.csv', header=True, index=False)
        
        wall_box = input_wall.bounding_box
        wall_grid = input_wall.grid
        
        def add_cc_offsets(brick, cc_overlaps):
            brick_index = BrickWall.compute_brick_index(brick, wall_box, wall_grid)
            cc_offset = np.uint64(brick_index * (max_brick_cc+1))
            
            # Don't overwrite zero voxels
            offset_cc_vol = np.where(brick.volume, brick.volume + cc_offset, np.uint64(0))
            cc_overlaps = cc_overlaps.copy()
            cc_overlaps.loc[(cc_overlaps['cc'] != 0), 'cc'] += np.uint64(cc_offset)
            
            # Append columns for brick location while we're at it.
            cc_overlaps['lz0'] = cc_overlaps['ly0'] = cc_overlaps['lx0'] = np.int32(0)
            cc_overlaps.loc[:, ['lz0', 'ly0', 'lx0']] = brick.logical_box[0]

            brick.compress()
            new_brick = Brick( brick.logical_box,
                               brick.physical_box,
                               offset_cc_vol,
                               location_id=brick.location_id,
                               compression=brick.compression )
            
            return new_brick, cc_overlaps

        # Now relabel each cc_brick so that label ids in different bricks never coincide
        offset_cc_results = bag_zip(cc_bricks, cc_overlaps).starmap(add_cc_offsets)
        offset_cc_results.persist()
        cc_bricks, cc_overlaps = offset_cc_results.unzip(2)

        # Extract halos.        
        # Note: No need to extract halos on all sides: outer-lower will overlap
        #       with inner-upper, which is good enough for computing CC.
        outer_halos = extract_halos(cc_bricks, input_wall.grid, 'outer', 'lower')
        inner_halos = extract_halos(cc_bricks, input_wall.grid, 'inner', 'upper')

        outer_halos_ddf = BrickWall.bricks_as_ddf(outer_halos, logical=False, physical=True, names='long')
        inner_halos_ddf = BrickWall.bricks_as_ddf(inner_halos, logical=False, physical=True, names='long')
        
        # Combine halo DFs along physical boxes, so that each outer halo is paired
        # with its overlapping partner (an inner halo, extracted from a different original brick).
        # Note: This is a pandas 'inner' merge, not to be confused with the 'inner' halos!
        combined_halos_ddf = outer_halos_ddf.merge(inner_halos_ddf, 'inner', PB_COLS, suffixes=['_outer', '_inner'])

        def find_pairwise_links(outer_brick, inner_brick):
            assert (outer_brick.physical_box == inner_brick.physical_box).all()

            # TODO: If this workflow is ever refactored into a set of utility functions,
            #       where each brick's segmentation might have been computed independently, 
            #       we'll probably want to permit the user to specify a minimum required
            #       overlap for neighboring objects to be considered 'linked'.

            table = pd.DataFrame({ 'cc_outer': outer_brick.volume.reshape(-1),
                                   'cc_inner': inner_brick.volume.reshape(-1) })
            table = table.drop_duplicates()
            #table = contingency_table(outer_brick.volume, inner_brick.volume).reset_index()
            #table.rename(columns={'left': 'cc_outer', 'right': 'cc_inner'}, inplace=True)

            outer_brick.compress()
            inner_brick.compress()

            # Omit label 0
            table = table.query('cc_outer != 0 and cc_inner != 0')
            return table

        def find_partition_links(combined_halos_df):
            tables = []
            for row in combined_halos_df.itertuples(index=False):
                table = find_pairwise_links(row.brick_outer, row.brick_inner)
                tables.append(table)

            if tables:
                return pd.concat(tables, ignore_index=True)
            else:
                return pd.DataFrame({'cc_outer': [], 'cc_inner': []}, dtype=np.uint64)
            
        links_meta = { 'cc_outer': np.uint64, 'cc_inner': np.uint64 }
        
        with Timer("Offsetting block CC and computing links", logger):
            links_df = combined_halos_ddf.map_partitions(find_partition_links, meta=links_meta).clear_divisions()
            links_df = links_df.compute()
            assert links_df.columns.tolist() == ['cc_outer', 'cc_inner'] 
            assert (links_df.dtypes == np.uint64).all()

        with Timer("Writing links_df.pkl", logger):
            pickle.dump(links_df, open('links_df.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        with Timer("Concatenating cc_overlaps", logger):
            cc_mapping_df = pd.concat(cc_overlaps.compute(), ignore_index=True)
            cc_mapping_df = cc_mapping_df[['lz0', 'ly0', 'lx0', 'orig', 'cc']]
            
        with Timer("Writing cc_mapping_df.pkl", logger):
            pickle.dump(cc_mapping_df, open('cc_mapping_df.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            
        #
        # Append columns for original labels
        #
        with Timer("Joining link CC column with original labels", logger):
            # The pandas way...
            #cc_mapping = cc_mapping_df.set_index('cc')['orig']
            #links_df = links_df.merge(cc_mapping, 'left', left_on='cc_outer', right_index=True)[['cc_outer', 'cc_inner', 'orig']]
            #links_df = links_df.merge(cc_mapping, 'left', left_on='cc_inner', right_index=True, suffixes=['_outer', '_inner'])

            # The LabelMapper way...
            assert (links_df.dtypes == np.uint64).all()
            assert (cc_mapping_df[['cc', 'orig']].dtypes == np.uint64).all()

            cc = cc_mapping_df['cc'].values
            cc_orig = cc_mapping_df['orig'].values
            cc_mapper = LabelMapper(cc, cc_orig)
            links_df['orig_outer'] = cc_mapper.apply(links_df['cc_outer'].values)
            links_df['orig_inner'] = cc_mapper.apply(links_df['cc_inner'].values)
            
            # If we know the input segmentation source is the same for every brick
            # (i.e. it comes from a pre-computed source, where the halos should exactly match),
            # Then this assertion is true. 
            # It will not be true if we ever change this workflow to a general connected
            # components workflow, where each block of segmentaiton might be generated
            # independently, and thus halos may not match exactly.
            assert links_df.eval('orig_outer == orig_inner').all(), \
                "Something is wrong -- either the halos are not aligned, or the mapping of CC->orig is wrong."

            links_df = links_df[['cc_outer', 'cc_inner', 'orig_outer', 'orig_inner']]
            assert (links_df.dtypes == np.uint64).all()

        with Timer("Writing links_df.pkl", logger):
            pickle.dump(links_df, open('links_df.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        with Timer("Dropping 'island' components", logger):
            # Before computing the CC of the whole graph, we can 
            # immediately discard any 'island' objects that only appear once.
            # That should make the CC operation run faster.

            # Note:
            #    Since our original segmentation blocks come from
            #    a common source (not computed independently), there is technically
            #    no need to also check for linked nodes, since by definition both sides
            #    of a link have the same original label.
            #    However, if we ever convert this code into a general connected components
            #    workflow, in which the segmentation blocks (including halos) might have
            #    been generated independently, then this additional check will be necessary.
            #    
            #     linked_nodes_orig = set(links_df['orig_outer'].values) | set(links_df['orig_inner'].values)
            #     node_df = cc_mapping_df.query(orig in @repeated_orig_labels or orig in @linked_nodes_orig')

            multiblock_rows = cc_mapping_df['orig'].duplicated(keep=False)
            node_df = cc_mapping_df.loc[multiblock_rows].copy()

        with Timer("Computing link CC", logger):
            # Compute connected components across all linked objects
            halo_links = links_df[['cc_outer', 'cc_inner']].values
            link_cc = connected_components_nonconsecutive(halo_links, node_df['cc'].values)
            node_df['link_cc'] = link_cc.astype(np.uint64)
            del halo_links, link_cc

        with Timer("Writing node_df_unfiltered.pkl", logger):
            pickle.dump(node_df, open('node_df_unfiltered.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        with Timer("Dropping unsplit objects", logger):
            # Original objects that ended up with only a single
            # component need not be relabeled. Drop them from the table.

            # Note:
            #    Since our original segmentation blocks come from
            #    a common source (not computed independently), there is technically
            #    no need to also check for merged original objects, since by definition
            #    both sides of a link have the same original label.
            #    However, if we ever convert this code into a general connected components
            #    workflow, in which the segmentation blocks (including halos) might have
            #    been generated independently, then we must also keep objects that were part
            #    of a linked_cc with multiple original components.
            #
            #      original_cc_counts = node_df.groupby('link_cc').agg({'orig': 'nunique'}).rename(columns={'orig': 'num_components'})
            #      final_cc_counts = node_df.groupby('orig').agg({'link_cc': 'nunique'}).rename(columns={'link_cc': 'num_components'})
            #      merged_orig_objects = original_cc_counts.query('num_components > 1').index #@UnusedVariable
            #      split_orig_objects = final_cc_counts.query('num_components > 1').index #@UnusedVariable
            #      node_df = node_df.query('orig in @split_orig_objects or link_cc in @merged_orig_objects').copy()

            final_cc_counts = node_df.groupby('orig').agg({'link_cc': 'nunique'}).rename(columns={'link_cc': 'num_components'}, copy=False)
            split_orig_objects = final_cc_counts.query('num_components > 1').index #@UnusedVariable
            node_df = node_df.query('orig in @split_orig_objects').copy()
            num_final_fragments = len(pd.unique(node_df['link_cc']))

        # Compute the final label for each cc
        # Start by determining where the final label range should start.
        next_label = self.determine_next_label(num_final_fragments, orig_maxes)
        link_ccs = pd.unique(node_df['link_cc'].values)

        with Timer("Computing final mapping", logger):
            # Map link_cc to final_cc
            mapper = LabelMapper(link_ccs, np.arange(next_label, next_label+len(link_ccs), dtype=np.uint64))
            node_df['final_cc'] = mapper.apply(node_df['link_cc'].values)
        
        with Timer("Writing node_df_final.pkl", logger):
            pickle.dump(node_df, open('node_df_final.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        if options["log-relabeled-objects"]:
            # This is mostly for convenient unit testing
            with Timer("Writing relabeled-objects.csv", logger):
                columns = {'final_cc': 'final_label', 'orig': 'orig_label'}
                csv_df = node_df[['final_cc', 'orig']].rename(columns=columns, copy=False).drop_duplicates()
                csv_df.to_csv('relabeled-objects.csv', index=False, header=True)
       
        #
        # Construct a Dask Dataframe holding the bricks we need to write.
        #
        wall_box = input_wall.bounding_box
        wall_grid = input_wall.grid

        with Timer("Preparing brick DataFrame", logging):
            def coords_and_bricks(orig_brick, cc_brick):
                assert (orig_brick.logical_box == cc_brick.logical_box).all()
                brick_index = BrickWall.compute_brick_index(orig_brick, wall_box, wall_grid)
                return (brick_index, orig_brick, cc_brick)
    
            dtypes = {'brick_index': np.int32, 'orig_brick': object, 'cc_brick': object}
            bricks_ddf = (bag_zip(input_wall.bricks, cc_bricks)
                            .starmap(coords_and_bricks)
                            .to_dataframe(dtypes))
    
            with Timer("Setting brick_index", logger):
                bricks_ddf = bricks_ddf.persist()
                bi = bricks_ddf['brick_index'].compute().tolist()
                assert bi == sorted(bi)
    
                bricks_ddf = bricks_ddf.set_index('brick_index', sorted=True)

        # The final mapping (node_df) might be too large to broadcast to all workers entirely.
        # We need to send only the relevant portion to each brick.
        with Timer("Preparing final mapping DataFrame", logger):
            class WrappedDf:
                """
                Trivial wrapper to allow us to store an entire
                DataFrame in every row of the following groupby()
                result."""
                def __init__(self, df):
                    self.df = df.copy()

                def __sizeof__(self):
                    return super().__sizeof__() + sys.getsizeof(self.df)
            
            with Timer("Grouping final CC mapping by brick", logger):
                grouped_mapping_df = (node_df[['lz0', 'ly0', 'lx0', 'cc', 'final_cc']]
                                        .groupby(['lz0', 'ly0', 'lx0'])
                                        .apply(WrappedDf)
                                        .rename('wrapped_brick_mapping_df')
                                        .reset_index())
    
            # We will need to merge according to brick location.
            # We could do that on [lz0,ly0,lx0], but a single-column merge will be faster,
            # so compute the brick_indexes and use that.
            brick_corners = grouped_mapping_df[['lz0', 'ly0', 'lx0']].values
            grouped_mapping_df['brick_index'] = BrickWall.compute_brick_indexes(brick_corners, wall_box, wall_grid)
            
            bi = grouped_mapping_df['brick_index'].tolist()
            assert bi == sorted(bi)
            
            grouped_mapping_df = grouped_mapping_df.set_index('brick_index')[['wrapped_brick_mapping_df']]
            grouped_mapping_ddf = ddf.from_pandas(grouped_mapping_df,
                                                  name='grouped_mapping',
                                                  npartitions=max(1, bricks_ddf.npartitions // 100))
            # Note:
            #   I've seen strange errors here from ddf.DataFrame.repartition() if all partitions are empty.
            #   If that's what you're seeing, make sure the input is reasonable.
            grouped_mapping_ddf = grouped_mapping_ddf.repartition(npartitions=bricks_ddf.npartitions)
            assert None not in grouped_mapping_ddf.divisions

        with Timer("Joining mapping and bricks", logger):
            # This merge associates each brick's part of the mapping with the correct row of bricks_ddf 
            bricks_ddf = bricks_ddf.merge(grouped_mapping_ddf, 'left', left_index=True, right_index=True)
            
            # We're done with these.
            del cc_bricks
            del input_wall

        def remap_cc_to_final(orig_brick, cc_brick, wrapped_brick_mapping_df):
            """
            Given an original brick and the corresponding CC brick,
            Relabel the CC brick according to the final label mapping,
            as provided in wrapped_brick_mapping_df.
            """
            assert (orig_brick.logical_box == cc_brick.logical_box).all()
            assert (orig_brick.physical_box == cc_brick.physical_box).all()

            # Check for NaN, which implies that the mapping for this
            # brick is empty (no objects to relabel).
            if isinstance(wrapped_brick_mapping_df, float):
                assert np.isnan(wrapped_brick_mapping_df)
                final_vol = orig_brick.volume
                orig_brick.compress()
            else:
                # Construct mapper from only the rows we need
                cc_labels = pd.unique(cc_brick.volume.reshape(-1)) # @UnusedVariable
                mapping = wrapped_brick_mapping_df.df.query('cc in @cc_labels')[['cc', 'final_cc']].values
                mapper = LabelMapper(*mapping.transpose())
    
                # Apply mapping to CC vol, writing zeros whereever the CC isn't mapped.
                final_vol = mapper.apply_with_default(cc_brick.volume, 0)
                
                # Overwrite zero voxels from the original segmentation.
                final_vol = np.where(final_vol, final_vol, orig_brick.volume)
            
                orig_brick.compress()
                cc_brick.compress()
            
            final_brick = Brick( orig_brick.logical_box,
                                 orig_brick.physical_box,
                                 final_vol,
                                 location_id=orig_brick.location_id,
                                 compression=orig_brick.compression )
            return final_brick

        collect_stats = options["compute-block-statistics"]

        bw = self.output_service.base_service.block_width
        if bw > 0:
            block_shape = 3*[self.output_service.base_service.block_width]
        else:
            block_shape = self.output_service.base_service.preferred_message_shape
         
        def write_brick(full_brick):
            brick = clip_to_logical(full_brick, False)
            
            # Don't re-compress; we're done with the brick entirely
            full_brick.destroy()

            vol = brick.volume
            brick.destroy()

            output_service.write_subvolume(vol, brick.physical_box[0], 0)
            if collect_stats:
                stats = block_stats_for_volume(block_shape, vol, brick.physical_box)
                return stats


        with Timer("Relabeling bricks and writing to output", logger):
            final_bricks = bricks_ddf.to_bag().starmap(remap_cc_to_final)
            del bricks_ddf
            all_stats = final_bricks.map(write_brick).compute()

        if collect_stats:
            with Timer("Writing block stats"):
                stats_df = pd.concat(all_stats, ignore_index=True)
                self.write_block_stats(stats_df)


    def init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        """
        input_config = self.config["input"]
        output_config = self.config["output"]
        mgr_config = self.config["resource-manager"]

        self.resource_mgr_client = ResourceManagerClient( mgr_config["server"], mgr_config["port"] )
        self.input_service = VolumeService.create_from_config( input_config, self.resource_mgr_client )

        # If we need to create a dvid instance for the output,
        # default to the same pyramid depth as the input
        if ("dvid" in input_config) and ("dvid" in output_config) and (output_config["dvid"]["creation-settings"]["max-scale"] == -1):
            info = fetch_instance_info(*self.input_service.base_service.instance_triple)
            pyramid_depth = info['Extended']['MaxDownresLevel']
            output_config["dvid"]["creation-settings"]["max-scale"] = pyramid_depth

        replace_default_entries(output_config["geometry"]["bounding-box"], self.input_service.bounding_box_zyx[:, ::-1])

        self.output_service = VolumeService.create_from_config( output_config, self.resource_mgr_client )
        assert isinstance( self.output_service, VolumeServiceWriter ), \
            "The output format you are attempting to use does not support writing"

        if isinstance(self.output_service.base_service, DvidVolumeService):
            if not self.output_service.base_service.supervoxels:
                raise RuntimeError("Can't write to a non-supervoxels output service.")

            if not self.output_service.base_service.disable_indexing:
                logger.warning("******************************************************************************")
                logger.warning("Your output config does not specify 'disable-indexing', which means DVID will "
                               "attempt to index all voxels as they are written to the volume. "
                               "For large volumes, this is NOT recommended!"
                               "(You should run a separate job to recompute the labelindex afterwards.)")
                logger.warning("******************************************************************************")

        logger.info(f"Output bounding box: {self.output_service.bounding_box_zyx[:,::-1].tolist()}")


    def init_brickwall(self, volume_service, subset_labels, roi):
        sbm = None

        if roi["name"]:
            base_service = volume_service.base_service

            if not roi["server"] or not roi["uuid"]:
                assert isinstance(base_service, DvidVolumeService), \
                    "Since you aren't using a DVID input source, you must specify the ROI server and uuid."

            roi["server"] = (roi["server"] or volume_service.server)
            roi["uuid"] = (roi["uuid"] or volume_service.uuid)

            scale = 0
            if isinstance(volume_service, ScaledVolumeService):
                scale = volume_service.scale_delta
                assert scale <= 5, \
                    "The 'roi' option doesn't support volumes downscaled beyond level 5"

            brick_shape = volume_service.preferred_message_shape
            assert not (brick_shape % 2**(5-scale)).any(), \
                "If using an ROI, select a brick shape that is divisible by 32"

            seg_box = volume_service.bounding_box_zyx
            seg_box = round_box(seg_box, 2**(5-scale))
            seg_box_s0 = seg_box * 2**scale
            seg_box_s5 = seg_box // 2**(5-scale)

            with Timer(f"Fetching mask for ROI '{roi}' ({seg_box_s0[:, ::-1].tolist()})", logger):
                roi_mask_s5, _ = fetch_roi(roi["server"], roi["uuid"], roi["name"], format='mask', mask_box=seg_box_s5)

            # SBM 'full-res' corresponds to the input service voxels, not necessarily scale-0.
            sbm = SparseBlockMask(roi_mask_s5, seg_box, 2**(5-scale))

        elif subset_labels:
            try:
                sbm = volume_service.sparse_block_mask_for_labels([*subset_labels])
                if ((sbm.box[1] - sbm.box[0]) == 0).any():
                    raise RuntimeError("Could not find sparse masks for any of the subset-labels")
            except NotImplementedError:
                sbm = None

        with Timer("Initializing BrickWall", logger):
            # Aim for 2 GB RDD partitions when loading segmentation
            GB = 2**30
            target_partition_size_voxels = 2 * GB // np.uint64().nbytes

            # Apply halo WHILE downloading the data.
            # TODO: Allow the user to configure whether or not the halo should
            #       be fetched from the outset, or added after the blocks are loaded.
            halo = self.config["connectedcomponents"]["halo"]
            brickwall = BrickWall.from_volume_service(volume_service, 0, None, self.client, target_partition_size_voxels, halo, sbm, compression='lz4_2x')

        return brickwall


    def determine_next_label(self, num_new_labels, orig_maxes):
        """
        Given a number of new labels we need to use for relabeling the volume,
        determine the first available label value we can use.
        
        In the case of DVID volumes, reserve the number of labels we need
        from the server (unless the user has overridden this via orig-max-label).
        
        If there is no user-specified value, and the output is not a DVID volume,
        Evaluate the given orig_maxes (a dask Bag), which were obtained from the
        input data, and use it to determine the next available label.
        """
        user_specified_max = self.config["connectedcomponents"]["orig-max-label"]
        if user_specified_max:
            next_label = user_specified_max+1
        
        elif isinstance(self.output_service.base_service, DvidVolumeService):
            server, uuid, instance = self.output_service.base_service.instance_triple

            # If the original maxlabel is less than one of values we'll be writing,
            # advance the instance to the new max before reserving yet more labels,
            # for our new segments.
            orig_max = orig_maxes.max().compute()
            maxlabel_output = fetch_maxlabel(server, uuid, instance)
            
            if maxlabel_output < orig_max:
                post_maxlabel(server, uuid, instance, maxlabel_output)
            
            # Now reserve new labels for our new fragments, which we'll be writing next.
            next_label, last_label = post_nextlabel(server, uuid, instance, num_new_labels)
            assert last_label+1 - next_label == num_new_labels

        else:
            next_label = orig_maxes.max().compute() + 1

        return np.uint64(next_label)


    def init_block_stats_file(self):
        stats_path = self.config["connectedcomponents"]["block-statistics-file"]
        if os.path.exists(stats_path):
            logger.warning(f"Block statistics already exists: {stats_path}")
            logger.warning(f"Will APPEND to the pre-existing statistics file.")
        elif stats_path.endswith('.csv'):
            # Initialize (just the header)
            template_df = pd.DataFrame(columns=list(BLOCK_STATS_DTYPES.keys()))
            template_df.to_csv(stats_path, index=False, header=True)
        elif stats_path.endswith('.h5'):
            # Initialize a 0-entry 1D array with the correct (structured) dtype
            with h5py.File(stats_path, 'w') as f:
                f.create_dataset('stats', shape=(0,), maxshape=(None,), chunks=True, dtype=list(BLOCK_STATS_DTYPES.items()))
        else:
            raise RuntimeError(f"Unknown file format: {stats_path}")
        

    def write_block_stats(self, stats_df):
        """
        Write the block stats.

        Args:
            slab_stats_df: DataFrame to be appended to the stats file,
                           with columns and dtypes matching BLOCK_STATS_DTYPES
        """
        self.init_block_stats_file()
        assert list(stats_df.columns) == list(BLOCK_STATS_DTYPES.keys())
        stats_path = self.config["connectedcomponents"]["block-statistics-file"]

        if stats_path.endswith('.csv'):
            stats_df.to_csv(stats_path, header=False, index=False, mode='a')

        elif stats_path.endswith('.h5'):
            with h5py.File(stats_path, 'a') as f:
                orig_len = len(f['stats'])
                new_len = orig_len + len(stats_df)
                f['stats'].resize((new_len,))
                f['stats'][orig_len:new_len] = stats_df.to_records()
        else:
            raise RuntimeError(f"Unknown file format: {stats_path}")


