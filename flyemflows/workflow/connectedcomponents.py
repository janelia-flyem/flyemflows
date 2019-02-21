import os
import copy
import logging

import numpy as np
import pandas as pd
import dask.bag as db
import skimage.measure as skm
from dask.bag import zip as bag_zip
from requests import HTTPError

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, Grid, SparseBlockMask, connected_components_nonconsecutive, apply_mask_for_labels
from neuclease.dvid import ( fetch_instance_info, fetch_sparsevol_coarse_via_labelindex, fetch_maxlabel,
                             post_maxlabel, post_nextlabel )

from dvidutils import LabelMapper

from ..util import replace_default_entries
from ..brick import Brick, BrickWall, extract_halos, PB_COLS, clip_to_logical
from ..volumes import VolumeService, VolumeServiceWriter, SegmentationVolumeSchema, DvidVolumeService
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
            "subset-labels": BodyListSchema,
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
            }
        }
    }

    ConnectedComponentsOptionsSchema["properties"]["subset-labels"]["description"] += (
        "If provided, only the listed labels will be analyzed for connected components analysis.\n"
        "Other labels will be left untouched in the results.\n")

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
        
        input_wall = self.init_brickwall(input_service, subset_labels)
        
        def brick_cc(brick):
            orig_vol = brick.volume
            orig_max = orig_vol.max()
            
            if subset_labels:
                apply_mask_for_labels(orig_vol.copy('C'), subset_labels, inplace=True)
            
            cc_vol = skm.label(orig_vol, background=0, connectivity=1)
            assert cc_vol.dtype == np.int64
            cc_vol = cc_vol.view(np.uint64)
            
            # Leave 0-pixels alone.
            cc_vol[orig_vol == 0] = np.uint64(0)
            
            cc_overlaps = pd.DataFrame({'orig': orig_vol.reshape(-1), 'cc': cc_vol.reshape(-1)})
            cc_overlaps.query('orig != 0 and cc != 0', inplace=True)
            cc_overlaps.drop_duplicates(inplace=True)

            cc_max = cc_overlaps['cc'].max()
            
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
        
        wall_box = input_wall.bounding_box
        wall_grid = input_wall.grid
        
        def add_cc_offsets(brick_and_cc_overlaps):
            brick, cc_overlaps = brick_and_cc_overlaps
            brick_index = BrickWall.compute_brick_index(brick, wall_box, wall_grid)
            cc_offset = np.uint64(brick_index * (max_brick_cc+1))
            
            # Don't overwrite zero voxels
            offset_cc_vol = np.where(brick.volume, brick.volume + cc_offset, np.uint64(0))
            cc_overlaps = cc_overlaps.copy()
            cc_overlaps.loc[(cc_overlaps['cc'] != 0), 'cc'] += np.uint64(cc_offset)
            
            new_brick = Brick( brick.logical_box,
                               brick.physical_box,
                               offset_cc_vol,
                               location_id=brick.location_id,
                               compression=brick.compression )
            
            return new_brick, cc_overlaps

        # Now relabel each cc_brick so that no labels overlap
        offset_cc_results = bag_zip(cc_bricks, cc_overlaps).map(add_cc_offsets)
        offset_cc_results.persist()
        cc_bricks, cc_overlaps = offset_cc_results.unzip(2)

        # Compute halos.        
        # Note: No need to compute ALL halos: outer-lower will overlap
        #       with inner-upper, which is good enough for computing CC.
        outer_halos = extract_halos(cc_bricks, input_wall.grid, 'outer', 'lower')
        inner_halos = extract_halos(cc_bricks, input_wall.grid, 'inner', 'upper')

        outer_halos_df = BrickWall.bricks_as_ddf(outer_halos)
        inner_halos_df = BrickWall.bricks_as_ddf(inner_halos)
        
        # Combine halo DFs along physical boxes, so that each outer halo is paired with it's partner (an inner halo).
        # This is an pandas 'inner' merge, not to be confused with the 'inner' halos!
        combined_halos_df = outer_halos_df.merge(inner_halos_df, 'inner', PB_COLS, suffixes=['_outer', '_inner'])

        def find_pairwise_links(outer_brick, inner_brick):
            assert (outer_brick.physical_box == inner_brick.physical_box).all()

            # TODO: If this workflow is ever refactored into a set of utility functions,
            #       where each brick's segmentation might have been computed independently, 
            #       we'll probably want to permit the user to specify a minimum required
            #       overlap for neighboring objects to be considered 'linked'.

            table = pd.DataFrame({'cc_outer': outer_brick.volume.reshape(-1), 'cc_inner': inner_brick.volume.reshape(-1)})
            table = table.drop_duplicates()
            #table = contingency_table(outer_brick.volume, inner_brick.volume).reset_index()
            #table.rename(columns={'left': 'cc_outer', 'right': 'cc_inner'}, inplace=True)

            # Omit label 0
            table.query('cc_outer != 0 and cc_inner != 0', inplace=True)
            return table

        def find_partition_links(combined_halos_df):
            tables = []
            for row in combined_halos_df.itertuples(index=False):
                table = find_pairwise_links(row.brick_outer, row.brick_inner)
                tables.append(table)

            return pd.concat(tables, ignore_index=True)
            
        links_meta = { 'cc_outer': np.uint64, 'cc_inner': np.uint64 }
        
        with Timer("Offsetting block CC and computing links", logger):
            links_df = combined_halos_df.map_partitions(find_partition_links, meta=links_meta).clear_divisions()
            links_df = links_df.compute()
            assert links_df.columns.tolist() == ['cc_outer', 'cc_inner'] 
            assert (links_df.dtypes == np.uint64).all()

        with Timer("Joining link CC column with original labels", logger):
            #
            # Append columns for original labels
            #
            cc_mapping_df = pd.concat(cc_overlaps.compute(), ignore_index=True)
            assert cc_mapping_df.columns.tolist() == ['orig', 'cc']
            links_df = links_df.merge(cc_mapping_df, 'left', left_on='cc_outer', right_on='cc')[['cc_outer', 'cc_inner', 'orig']]
            links_df = links_df.merge(cc_mapping_df, 'left', left_on='cc_inner', right_on='cc', suffixes=['_outer', '_inner'])
            links_df = links_df[['cc_outer', 'cc_inner', 'orig_outer', 'orig_inner']]
            assert (links_df.dtypes == np.uint64).all()

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

            repeated_orig_labels = (cc_mapping_df['orig'].value_counts() > 2).index #@UnusedVariable
            node_df = cc_mapping_df.query('orig in @repeated_orig_labels')

        with Timer("Computing link CC", logger):
            # Compute connected components across all linked objects
            node_df['link_cc'] = connected_components_nonconsecutive(links_df[['cc_outer', 'cc_inner']].values, node_df['cc'].values)

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
        
        csv_df = node_df[['final_cc', 'orig']].rename(columns={'final_cc': 'final_label', 'orig': 'orig_label'}, copy=False).drop_duplicates()
        csv_df.to_csv('relabeled-objects.csv', index=False, header=True)

        # Delete unnecessary columns before broadcasting this DF to all workers.
        del node_df['orig']
        del node_df['link_cc']

        def remap_cc_to_final(orig_and_cc):
            """
            Given an original brick and the corresponding CC brick,
            Relabel the CC brick according to the final label mapping.
            
            Note:
                The final mapping (node_df) is implicitly broadcasted to all workers.
                Hopefully it isn't prohibitively large.
            """
            orig_brick, cc_brick = orig_and_cc
            assert (orig_brick.logical_box == cc_brick.logical_box).all()
            assert (orig_brick.physical_box == cc_brick.physical_box).all()
            
            # Construct mapper from only the rows we need
            cc_labels = pd.unique(cc_brick.volume.reshape(-1)) # @UnusedVariable
            mapping = node_df.query('cc in @cc_labels')[['cc', 'final_cc']].values
            mapper = LabelMapper(*mapping.transpose())

            # Apply mapping to CC vol, writing zeros whereever the CC isn't mapped.
            final_vol = mapper.apply_with_default(cc_brick.volume, 0)
            
            # Overwrite zero voxels from the original segmentation.
            final_vol = np.where(final_vol, final_vol, orig_brick.volume)
            
            final_brick = Brick( orig_brick.logical_box,
                                 orig_brick.physical_box,
                                 final_vol,
                                 location_id=orig_brick.location_id,
                                 compression=orig_brick.compression )
            return final_brick        

        def write_brick(brick):
            brick = clip_to_logical(brick, False)
            output_service.write_subvolume(brick.volume, brick.physical_box[0], 0)

        with Timer("Relabeling bricks and writing to output", logger):
            final_bricks = bag_zip(input_wall.bricks, cc_bricks).map(remap_cc_to_final)
            final_bricks.map(write_brick).compute()


    def init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        """
        input_config = self.config["input"]
        output_config = self.config["output"]
        mgr_config = self.config["resource-manager"]

        self.resource_mgr_client = ResourceManagerClient( mgr_config["server"], mgr_config["port"] )
        self.input_service = VolumeService.create_from_config( input_config, os.getcwd(), self.resource_mgr_client )

        # If we need to create a dvid instance for the output,
        # default to the same pyramid depth as the input
        if ("dvid" in input_config) and ("dvid" in output_config) and (output_config["dvid"]["creation-settings"]["max-scale"] == -1):
            info = fetch_instance_info(*self.input_service.instance_triple)
            pyramid_depth = info['Extended']['MaxDownresLevel']
            output_config["dvid"]["creation-settings"]["max-scale"] = pyramid_depth

        replace_default_entries(output_config["geometry"]["bounding-box"], self.input_service.bounding_box_zyx[:, ::-1])
        self.output_service = VolumeService.create_from_config( output_config, os.getcwd(), self.resource_mgr_client )
        assert isinstance( self.output_service, VolumeServiceWriter ), \
            "The output format you are attempting to use does not support writing"

        if isinstance(self.output_service.base_service, DvidVolumeService):
            if not self.output_service.supervoxels:
                raise RuntimeError("Can't write to a non-supervoxels output service.")
            
            if not self.output_service.disable_indexing:
                logger.warning("******************************************************************************")
                logger.warning("Your output config does not specify 'disable-indexing', which means DVID will "
                               "attempt to index all voxels as they are written to the volume. "
                               "For large volumes, this is NOT recommended!"
                               "(You should run a separate job to recompute the labelindex afterwards.)")
                logger.warning("******************************************************************************")

        logger.info(f"Output bounding box: {self.output_service.bounding_box_zyx[:,::-1].tolist()}")
        

    def init_brickwall(self, volume_service, subset_labels):
        halo = self.config["connectedcomponents"]["halo"]
        sbm = self.init_sparseblockmask(volume_service, subset_labels)
            
        with Timer("Initializing BrickWall", logger):
            # Aim for 2 GB RDD partitions when loading segmentation
            GB = 2**30
            target_partition_size_voxels = 2 * GB // np.uint64().nbytes
            brickwall = BrickWall.from_volume_service(volume_service, 0, None, self.client, target_partition_size_voxels, 0, sbm)

            # Apply halo after downloading the data.
            # TODO: Allow the user to configure whether or not the halo should
            #       be fetched from the outset, or added after the blocks are loaded.
            overlapping_grid = Grid(brickwall.grid.block_shape, halo=halo)
            brickwall = brickwall.realign_to_new_grid(overlapping_grid)

        return brickwall


    def init_sparseblockmask(self, volume_service, labels):
        if not isinstance(volume_service.base_service, DvidVolumeService):
            return None
        
        if not isinstance(volume_service, DvidVolumeService):
            logger.warning("Fetching all bricks in the volume -- currently, sparse fetching only "
                           "works for raw dvid volumes (not scaled or transposed ones)")
            return None
    
        if len(labels) == 0:
            return None

        labels = set(labels)
        is_supervoxels = volume_service.supervoxels
        brick_shape = volume_service.preferred_message_shape
        blocks_per_brick = brick_shape // 64

        mgr = self.resource_mgr_client
        def fetch_coarse_and_divide(label):
            """
            Fetch the coarse sparsevol for the label and return a pair of
            (label, coords), where coords is a record array
            with columns [z,y,x,label], and z,y,x have been converted to
            brick indices (not block indicies).
            """
            try:
                with mgr.access_context(volume_service.server, True, 1, 1):
                    coords = fetch_sparsevol_coarse_via_labelindex(*volume_service.instance_triple, label, is_supervoxels)
                    coords //= blocks_per_brick
                    coords = pd.DataFrame(coords, columns=['z', 'y', 'x'], dtype=np.int32).drop_duplicates()
                    coords['label'] = np.uint64(label)
                return (label, coords.to_records(index=False))
            except HTTPError as ex:
                if (ex.response is not None and ex.response.status_code == 404):
                    return (label, None)
                raise

        with Timer(f"Fetching coarse sparsevols for {len(labels)} labels", logger=logger):
            labels = db.from_sequence(labels)
            labels_and_coords = labels.map(fetch_coarse_and_divide).compute()

        bad_labels = []
        for label, coords in labels_and_coords:
            if coords is None:
                bad_labels.append(label)
    
        if bad_labels:
            logger.warning(f"Could not obtain coarse sparsevol for {len(bad_labels)} labels: {bad_labels}")
            labels_and_coords = list( filter(lambda k_v: k_v[1] is not None, labels_and_coords) )
    
        all_coords = np.concatenate( [kv[1] for kv in labels_and_coords] )
        coords_df = pd.DataFrame( all_coords )
        coords_df.drop_duplicates(['z', 'y', 'x'], inplace=True)
        coords = coords_df[['z', 'y', 'x']].values
        
        min_coord = coords.min(axis=0)
        max_coord = coords.max(axis=0)
        mask_box = np.array((min_coord, 1+max_coord))
        mask_shape = mask_box[1] - mask_box[0]
        mask = np.zeros(mask_shape, dtype=bool)

        coords -= mask_box[0]
        mask[(*coords.transpose(),)] = True

        sbm = SparseBlockMask(mask, brick_shape*mask_box, brick_shape)
        return sbm


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

