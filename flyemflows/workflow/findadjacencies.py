import os
import copy
import logging
from itertools import combinations

import vigra
import numpy as np
import pandas as pd
from dask.delayed import delayed

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import (Timer, swap_df_cols, approximate_closest_approach, SparseBlockMask,
                            connected_components_nonconsecutive, apply_mask_for_labels)

from ilastikrag import Rag
from dvidutils import LabelMapper

from ..util import stdout_redirected
from ..brick import BrickWall
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService
from .util.config_helpers import BodyListSchema, load_body_list, LabelGroupSchema, load_label_groups
from . import Workflow

logger = logging.getLogger(__name__)

EDGE_TABLE_TYPES = {
    'label_a': np.uint64, 'label_b': np.uint64,
    'za': np.int32, 'ya': np.int32, 'xa': np.int32,
    'zb': np.int32, 'yb': np.int32, 'xb': np.int32,
    'distance': np.float32,
    'edge_area': np.int32
}

EDGE_TABLE_COLS = list(EDGE_TABLE_TYPES.keys())


class FindAdjacencies(Workflow):
    """
    Workflow to find all adjacent bodies in a segmentation volume,
    and output a coordinate for each that lies along the adjacency
    boundary, and somewhat near the centroid of the boundary.

    Additionally, this workflow can find near-adjacent "edges",
    for specified groups of labels.  In that case, the emitted coordinates
    are guaranteed to reside in the labels of interest, and reside close to
    the points of "closest approach" between the two listed bodies within
    the brick in which the bodies were analyzed.

    Note: This workflow performs well on medium-sized volumes,
          or on large volumes when filtering the set of edges using either
          the restrict-bodies or restrict-edges settings in the config.
          It will not perform well on an unfiltered hemibrain-sized volume,
          which would result in billions of edges.

    TODO: For VERY large (10M-100M) label subsets, the time required to select the sparse
          block set is really long (and requires a lot of RAM on the driver).
          There should be a config option to simply skip the label-based sparse,
          block determination, but maybe instead allow the user to specify an ROI
          (avoid downloading the whole bounding-box).


    More notes about the "find-closest" stage
    =========================================

    First, this workflow finds the actually adjacent label pairs (i.e. touching objects).
    But if the objects in a group cannot be linked together into a single connected
    component via the touching adjacencies alone, it may be desirable to "try harder",
    and search for "almost-touching" adjacencies that might link all objects in a group
    together, into a single connected component.  It is NOT our goal to find ALL
    "almost-touching" pairs in the group. The goal is merely to find "enough" links that
    all objects in a group can be SOMEHOW linked into a single connected component.

    For instance, consider the following segmentation, in which the user has listed
    labels [1,2,3,4,5] as belonging to the same group, despite the fact that they
    don't all touch:

       11111111
       11  222   4444
          33333   555

    After the first stage, the following exact (touching) adjacencies will be found:

        [1,2], [2,3], [4,5]

    With only those edges, the group contains two components: {1,2,3} and {4,5}.
    If the 'find-closest' feature is requested, then a second stage of analysis is performed,
    searching for some way to link the two components of the group. The only potential edges
    to consider are the ones that span from one component to another:

        [1,4], [1,5], [2,4], [2,5], [3,4], [3,5]

    So, no attempt is made to find "almost touching" edges for nodes that already reside in the
    same component. Hence, the pair [1,3] is not considered in the second stage. Although those
    two objects might come really close to each other (maybe even closer than the pairs we ARE
    considering), that link wouldn't help us reduce the number of connected components, so it
    is ignored.

    Furthermore, not all of the potential "almost touching" edges will necessarily be found.
    Heuristics are used to find at least one of them, but no attempt is made to find ALL of them.
    Again, the goal is just to find enough links to unify all of the objects into a single component
    (one component per group), if possible.
    """

    FindAdjacenciesOptionsSchema = \
    {
        "type": "object",
        "description": "Settings specific to the FindAdjacencies workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "halo": {
                "description": "How much overlapping context between bricks in the grid (in voxels)\n"
                               "If you leave it as 0, then even direct adjacencies can be missed at the brick boundaries.\n"
                               "If you set it to 1, all direct adjacencies will be found.\n"
                               "Bigger halos than 1 are only useful if you are using 'find-closest'.\n"
                               "(But even then, you may not capture everything you're looking for.)\n",
                "type": "integer",
                "minValue": 0,
                "default": 0
            },
            "subset-label-groups": LabelGroupSchema,
            "subset-labels": {
                **BodyListSchema,
                "description": "If provided, this list will be used to limit the set adjacencies returned.\n"
                               "See 'subset-labels-requirement'\n",
            },
            "subset-labels-requirement": {
                "description": "When using subset-labels, use this setting to specify whether\n"
                               "each edge must include 1 or 2 of the listed labels.\n",
                "type": "integer",
                "minValue": 1,
                "maxValue": 2,
                "default": 2
            },
            "subset-edges": {
                "description": "Discard all adjacencies that are not included in this list of edges,\n"
                               "given as a CSV with columns: label_a,label_b.\n",
                "type": "string",
                "default": ""
            },
            "find-closest-using-scale": {
                "description": "For body pairs that do not physically touch,\n"
                               "find the points at which they come closest to each other, if this setting is not null.\n"
                               "For perfect accuracy, set this to 0 (i.e. scale 0).\n"
                               "For faster performance, set this to a higher scale (1,2,3, etc.)\n"
                               "The results will be approximate (not necessarily the exact closest points, or even on the object borders)\n"
                               "but the resulting coordinates are still guaranteed to fall on the objects of interest.\n"
                               "(Only permitted when using a subset option.)\n",
                "oneOf": [{"type": "integer"}, {"type": "null"}],
                "default": None
            },
            "nudge-points-apart": {
                "description": "Optionally 'nudge' the points apart from eachother, away from the\n"
                               "exact edge boundary and slightly towards the interior of each object.\n"
                               "Note: the 'distance' associated with the edge is NOT updated to reflect the 'nudged' locations.\n",
                "type": "boolean",
                "default": False
            },
            "cc-distance-threshold": {
                "description": "When computing the connected components, don't use edges that exceed this distance threshold.\n"
                               "A threshold of 1.0 indicates that only direct adjacencies should be used.\n"
                               "Edges above this threshold will still be included in the results, but they will be marked with a 'cc' column of -1.\n",
                "type": "number",
                "default": 1.0
            },
            "output-table": {
                "description": "Results file.  Must be .csv for now, and must contain at least columns x,y,z",
                "type": "string",
                "default": "adjacencies.csv"
            }
        }
    }
    
    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,
        "findadjacencies": FindAdjacenciesOptionsSchema
    })


    @classmethod
    def schema(cls):
        return FindAdjacencies.Schema


    def _sanitize_config(self):
        options = self.config["findadjacencies"]
        subset_requirement = options["subset-labels-requirement"]
        find_closest_using_scale = options["find-closest-using-scale"]

        num_subsets = sum([ bool(options["subset-labels"]),
                            bool(options["subset-edges"]),
                            bool(options["subset-label-groups"]) ])
        if num_subsets > 1:
            raise RuntimeError("You cannot specify more than one subset mechanism. "
                               "Provide either subset-labels, subset-edges, or subset-label-groups.")


        if find_closest_using_scale is not None and subset_requirement != 2:
            raise RuntimeError("Can't use find-closest-using-scale unless subset-requirement == 2")

        assert subset_requirement == 2, \
            "FIXME: subset-requirement other than 2 is not currently supported."

    def _load_label_groups(self, volume_service):
        options = self.config["findadjacencies"]

        subset_edges = np.zeros((0,2), np.uint64)
        if options["subset-edges"]:
            subset_edges = pd.read_csv(options["subset-edges"], dtype=np.uint64, header=0, names=['label_a', 'label_b']).values
            subset_edges.sort(axis=1)
        subset_edges = pd.DataFrame(subset_edges, columns=['label_a', 'label_b'], dtype=np.uint64)
        subset_edges.query('label_a != label_b', inplace=True) # drop invalid
        subset_edges.drop_duplicates(inplace=True)

        subset_groups = load_label_groups(options["subset-label-groups"])
        assert subset_groups.duplicated().sum() == 0

        is_supervoxels = ( isinstance(volume_service.base_service, DvidVolumeService)
                           and volume_service.base_service.supervoxels )
        subset_labels = load_body_list(options["subset-labels"], is_supervoxels)

        # (sanitize_config guarantees that only one of the subset options is selected.)        
        if len(subset_edges) > 0:
            label_pairs = np.asarray(subset_edges)
            subset_groups = pd.DataFrame({'label': label_pairs.reshape(-1)})
            subset_groups['group'] = np.arange(label_pairs.size, dtype=np.uint32) // 2

        if len(subset_labels) > 0:
            subset_groups = pd.DataFrame({'label': subset_labels, 'group': np.uint32(1)})

        if len(subset_groups) == 0:
            # No subset -- this is probably an error.
            # (Currently, this workflow is intended for finding only sparse adjacencies.)
            msg = "Your config does not specify any subset of labels to find adjacencies for, or that subset is empty."
            raise RuntimeError(msg)

        subset_requirement = options["subset-labels-requirement"]
        if len(subset_groups) == 1 and subset_requirement == 2:
            raise RuntimeError("Only one body was listed in subset-bodies.  No edges would be found!")

        return subset_groups

    def execute(self):
        self._sanitize_config()
        input_config = self.config["input"]
        options = self.config["findadjacencies"]
        resource_config = self.config["resource-manager"]
        subset_requirement = options["subset-labels-requirement"]
        find_closest_using_scale = options["find-closest-using-scale"]
        nudge = options["nudge-points-apart"]

        self.resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])
        volume_service = VolumeService.create_from_config(input_config, self.resource_mgr_client)

        subset_groups = self._load_label_groups(volume_service)
        subset_groups["group_cc"] = np.arange(len(subset_groups), dtype=np.uint32)

        brickwall = self.init_brickwall(volume_service, subset_groups)

        with Timer("Finding direct adjacencies", logger):
            def find_adj(brick, sg):
                # FIXME: Instead of broadcasting the entire subset_groups to all tasks,
                #        it might be better to distribute each brick's rows.
                #        (But that will involve a dask join step...)
                #        The subset_groups should generally be under 1 GB, anyway...
                edges = find_edges_in_brick(brick, None, sg, subset_requirement, nudge)
                brick.compress()
                return edges
            adjacent_edge_tables = brickwall.bricks.map(find_adj, sg=delayed(subset_groups)).compute()

        with Timer("Combining/filtering direct adjacencies", logger):
            adjacent_edge_tables = list(filter(lambda t: t is not None, adjacent_edge_tables))
            if len(adjacent_edge_tables) == 0:
                all_adjacent_edges_df = pd.DataFrame([], columns=EDGE_TABLE_COLS).astype(EDGE_TABLE_TYPES)
                best_adjacent_edges_df = all_adjacent_edges_df
                best_adjacent_edges_df['group'] = np.zeros((0,), dtype=np.uint32)
                best_adjacent_edges_df['group_cc'] = np.zeros((0,), dtype=np.int32)
            else:
                all_adjacent_edges_df = pd.concat(adjacent_edge_tables, ignore_index=True)
                best_adjacent_edges_df = select_central_edges(all_adjacent_edges_df, ['za', 'ya', 'xa'])
                best_adjacent_edges_df, subset_groups = append_group_ccs(best_adjacent_edges_df, subset_groups, None)

        logger.info(f"Found {len(all_adjacent_edges_df)} direct intra-group adjacencies ({len(best_adjacent_edges_df)} after selection)")
        np.save('all-adjacent-brick-edges-for-debug.npy', all_adjacent_edges_df.to_records(index=False))
        np.save('best-adjacent-brick-edges-for-debug.npy', best_adjacent_edges_df.to_records(index=False))
        np.save('subset-group-ccs-for-debug.npy', subset_groups.to_records(index=False))

        # No non-adjacencies by default
        all_nonadjacent_edges_df = pd.DataFrame([], columns=EDGE_TABLE_COLS).astype(EDGE_TABLE_TYPES)
        best_nonadjacent_edges_df = all_nonadjacent_edges_df

        if find_closest_using_scale is not None:
            with Timer("Finding closest approaches", logger):
                def find_closest(brick, sg):
                    edges = find_edges_in_brick(brick, find_closest_using_scale, sg, subset_requirement, nudge)
                    brick.compress()
                    return edges
                nonadjacent_edge_tables = brickwall.bricks.map(find_closest, sg=delayed(subset_groups)).compute()

            with Timer("Combining closest approaches", logger):
                nonadjacent_edge_tables = list(filter(lambda t: t is not None, nonadjacent_edge_tables))
                if len(nonadjacent_edge_tables) != 0:
                    all_nonadjacent_edges_df = pd.concat(nonadjacent_edge_tables, ignore_index=True)
                    best_nonadjacent_edges_df = select_closest_edges(all_nonadjacent_edges_df)

        np.save('all-nonadjacent-brick-edges-for-debug.npy', all_nonadjacent_edges_df.to_records(index=False))
        np.save('best-nonadjacent-brick-edges-for-debug.npy', best_nonadjacent_edges_df.to_records(index=False))

        with Timer("Sorting edges", logger):
            del best_adjacent_edges_df['group']
            del best_adjacent_edges_df['group_cc']
            final_edges_df = pd.concat((best_adjacent_edges_df, best_nonadjacent_edges_df), ignore_index=True)
            # This sort isn't necessary, but makes it easier to diff
            # the results of one execution vs another, for debugging.
            final_edges_df.sort_values(list(final_edges_df.columns), inplace=True)

        # The reason this assertion holds is that the subset_groups table contains a group_cc
        # column, and that column is used to avoid looking for "closest approach" edges between
        # nodes that have already been found to be connected via direct adjacencies.
        # See _find_closest_approaches() for details.
        assert len(best_adjacent_edges_df.merge(best_nonadjacent_edges_df, 'inner', ['label_a', 'label_b'])) == 0, \
            "The sets of adjacent and non-adjacent edges were supposed to be completely disjoint."

        final_edges_df, subset_groups = append_group_ccs(final_edges_df, subset_groups, options["cc-distance-threshold"])

        with Timer("Writing edges", logger):
            final_edges_df.to_csv(options["output-table"], header=True, index=False)

            # Save in numpy format, too.
            npy_path = options["output-table"][:-4] + '.npy'
            np.save(npy_path, final_edges_df.to_records(index=False))


    def init_brickwall(self, volume_service, subset_groups):
        try:
            brick_coords_df = volume_service.sparse_brick_coords_for_label_groups(subset_groups)
            np.save('brick-coords.npy', brick_coords_df.to_records(index=False))

            brick_shape = volume_service.preferred_message_shape
            brick_indexes = brick_coords_df[['z', 'y', 'x']].values // brick_shape
            sbm = SparseBlockMask.create_from_lowres_coords(brick_indexes, brick_shape)
        except NotImplementedError:
            logger.warning("The volume service does not support sparse fetching.  All bricks will be analyzed.")
            sbm = None
            
        with Timer("Initializing BrickWall", logger):
            # Aim for 2 GB RDD partitions when loading segmentation
            GB = 2**30
            target_partition_size_voxels = 2 * GB // np.uint64().nbytes
            
            # Apply halo WHILE downloading the data.
            # TODO: Allow the user to configure whether or not the halo should
            #       be fetched from the outset, or added after the blocks are loaded.
            halo = self.config["findadjacencies"]["halo"]
            brickwall = BrickWall.from_volume_service(volume_service, 0, None, self.client, target_partition_size_voxels, halo, sbm, compression='lz4_2x')

        return brickwall


def append_group_col(edges_df, subset_groups):
    """
    For the given list of edges, append a 'group' ID to each row.
    An edge belongs to a group if the group contains both of its endpoints.
    If an edge belongs to multiple groups, the edge is duplicated so
    it can be listed with multiple group values.
    Edges whose endpoints do not have any groups in common are omitted from the output.

    Duplicate label pairs are permitted, in which case they will also appear duplicated
    in the output (with the other columns preserved).  As described above, they will be
    doubly-duplicated if the edge belongs to multiple groups.
    This is useful if the same label pair appears multiple times, but with different
    auxilliary columns (e.g. two labels are adjacent at multiple sites).

    Note:
        There is a test for this function in test_findadjacencies.py
    """
    subset_groups = subset_groups[['label', 'group']]

    # Tag rows with a unique ID, which we'll use in the final inner-merge below
    # to avoid double-duplications in the case of duplicated label pairs in which
    # one or more of the labels is involved in multiple groups.
    edges_df['row'] = np.arange(len(edges_df), dtype=int)

    # Assign label_a groups, (duplicating edges as necessary if label_a belongs to multiple groups)
    edges_a_df = edges_df.merge(subset_groups.rename(columns={'label': 'label_a'}, copy=False), 'left', on='label_a')

    # Assign label_b groups, (duplicating edges as necessary if label_b belongs to multiple groups)
    edges_b_df = edges_df[['label_a', 'label_b', 'row']].merge(subset_groups.rename(columns={'label': 'label_b'}, copy=False), 'left', on='label_b')

    # Keep rows that have matching groups on both sides
    edges_df = edges_a_df.merge(edges_b_df, 'inner', ['label_a', 'label_b', 'row', 'group'])

    # Put group first, and sort by group
    cols = edges_df.columns.tolist()
    cols.remove('group')
    cols.insert(0, 'group')
    edges_df = edges_df[cols]
    del edges_df['row']

    return edges_df


def append_group_ccs(edges_df, subset_groups, max_distance=None):
    """
    For the given edges_df, assign a group to each edge
    (duplicating edges if they belong to multiple groups),
    and return the cc id as a new column 'group_cc'.

    The CC operation is performed on all groups at once,
    using disjoint sets of node IDs for every group.
    Thus, the CC ids for each group do NOT start at 1.
    Rather, the values in group_cc are arbitrary and
    not even consecutive.

    max_distance:
        If provided, exclude edges that exceed this distance from
        the CC computation (but include them in the resulting dataframe).
        For such excluded edges, group_cc == -1.
    """
    with Timer("Computing group_cc", logger):
        edges_df = append_group_col(edges_df, subset_groups)

        # Assign a unique id for every label/group combination,
        # so we can run CC on the whole set at once.
        # Labels that appear more than once (in different groups)
        # will be treated as independent nodes,
        # and there will be no edges between groups.
        #
        # Note: Assigning node IDs this way assumes subset-requirement == 2
        subset_groups = subset_groups[['label', 'group']].copy()
        assert subset_groups.duplicated().sum() == 0
        subset_groups['node_id'] = subset_groups.index.astype(np.uint32)

        # Append columns for [node_id_a, node_id_b]
        sg_a = subset_groups.rename(columns={'label': 'label_a'}, copy=False)
        sg_b = subset_groups.rename(columns={'label': 'label_b'}, copy=False)
        edges_df = edges_df.merge(sg_a, 'left', on=['label_a', 'group'])
        edges_df = edges_df.merge(sg_b, 'left', on=['label_b', 'group'], suffixes=['_a', '_b'])

        # Drop edges that are too distant to consider for CC
        if max_distance is None:
            thresholded_edges = edges_df[['node_id_a', 'node_id_b']].values
        else:
            thresholded_edges = edges_df.query('distance <= @max_distance')[['node_id_a', 'node_id_b']].values

        # Compute CC on the entire edge set, yielding a unique id for every CC in each group
        group_cc = 1 + connected_components_nonconsecutive(thresholded_edges, subset_groups['node_id'].values)
        subset_groups['group_cc'] = group_cc.astype(np.int32)

        # Append group_cc to every row.
        # All edges we actually used will have the same group_cc for node_id_a/node_id_b,
        # so just use node_id_a as the lookup.
        edges_df = edges_df.merge(subset_groups[['node_id', 'group_cc']], 'left', left_on='node_id_a', right_on='node_id')
        edges_df = edges_df.drop(['node_id_a', 'node_id_b', 'node_id'], axis=1)

        # But edges that were NOT used in the CC computation might span between different components.
        # For those rows, group_cc has no valid value.  Set group_cc to -1.
        edges_df['group_cc'] = edges_df['group_cc'].astype(np.int32)
        edges_df.loc[edges_df['distance'] > max_distance, 'group_cc'] = np.int32(-1)
        return edges_df, subset_groups


def find_edges_in_brick(brick, closest_scale=None, subset_groups=[], subset_requirement=2, nudge=False):
    """
    Find all pairs of adjacent labels in the given brick,
    and find the central-most point along the edge between them.
    
    (Edges to/from label 0 are discarded.)
    
    If closest_scale is not None, then non-adjacent pairs will be considered,
    according to a particular heuristic to decide which pairs to consider.
    
    Args:
        brick:
            A Brick to analyze
        
        closest_scale:
            If None, then consider direct (touching) adjacencies only.
            If not-None, then non-direct "adjacencies" (i.e. close-but-not-touching) are found.
            In that case `closest_scale` should be an integer >=0 indicating the scale at which
            the analysis will be performed.
            Higher scales are faster, but less precise.
            See ``neuclease.util.approximate_closest_approach`` for more information.
        
        subset_groups:
            A DataFrame with columns [label, group].  Only the given labels will be analyzed
            for adjacencies.  Furthermore, edges (pairs) will only be returned if both labels
            in the edge are from the same group.
            
        subset_requirement:
            Whether or not both labels in each edge must be in subset_groups, or only one in each edge.
            (Currently, subset_requirement must be 2.)
        
    Returns:
        If the brick contains no edges at all (other than edges to label 0), return None.
        
        Otherwise, returns pd.DataFrame with columns:
            [label_a, label_b, forwardness, z, y, x, axis, edge_area, distance]. # fixme
        
        where label_a < label_b,
        'axis' indicates which axis the edge crosses at the chosen coordinate,
        
        (z,y,x) is always given as the coordinate to the left/above/front of the edge
        (depending on the axis).
        
        If 'forwardness' is True, then the given coordinate falls on label_a and
        label_b is one voxel "after" it (to the right/below/behind the coordinate).
        Otherwise, the coordinate falls on label_b, and label_a is "after".
        
        And 'edge_area' is the total count of the voxels touching both labels.
    """
    # Profiling indicates that query('... in ...') spends
    # most of its time in np.unique, believe it or not.
    # After looking at the implementation, I think it might help a
    # little if we sort the array first.
    brick_labels = np.sort(pd.unique(brick.volume.reshape(-1)))
    if (len(brick_labels) == 1) or (len(brick_labels) == 2 and (0 in brick_labels)):
        return None # brick is solid -- no possible edges

    # Drop labels that aren't even present
    subset_groups = subset_groups.query('label in @brick_labels').copy()

    # Drop groups that don't have enough members (usually 2) in this brick.
    group_counts = subset_groups['group'].value_counts()
    _kept_groups = group_counts.loc[(group_counts >= subset_requirement)].index
    subset_groups = subset_groups.query('group in @_kept_groups').copy()

    if len(subset_groups) == 0:
        return None # No possible edges to find in this brick.

    # Contruct a mapper that includes only the labels we'll keep.
    # (Other labels will be mapped to 0).
    # Also, the mapper converts to uint32 (required by _find_and_select_central_edges,
    # but also just good for RAM reasons).
    kept_labels = np.sort(np.unique(subset_groups['label'].values))
    remapped_kept_labels = np.arange(1, len(kept_labels)+1, dtype=np.uint32)
    mapper = LabelMapper(kept_labels, remapped_kept_labels)
    reverse_mapper = LabelMapper(remapped_kept_labels, kept_labels)

    # Construct RAG -- finds all edges in the volume, on a per-pixel basis.
    remapped_volume = mapper.apply_with_default(brick.volume, 0)
    brick.compress()
    remapped_subset_groups = subset_groups.copy()
    remapped_subset_groups['label'] = mapper.apply(subset_groups['label'].values)

    try:
        if closest_scale is None:
            best_edges_df = _find_and_select_central_edges(remapped_volume, remapped_subset_groups, subset_requirement)
        else:
            best_edges_df = _find_closest_approaches(remapped_volume, closest_scale, remapped_subset_groups)
    except:
        brick_name = f"{brick.logical_box[:,::-1].tolist()}"
        np.save(f'problematic-remapped-brick-{brick_name}.npy', remapped_volume)
        logger.error(f"Error in brick (XYZ): {brick_name}") # This will appear in the worker log.
        raise

    if best_edges_df is None:
        return None

    if nudge:
        # sign of direction from a -> b
        s = np.sign(
              best_edges_df[['zb', 'yb', 'xb']].values.astype(np.int32)
            - best_edges_df[['za', 'ya', 'xa']].values.astype(np.int32)
        )

        # Nudge by a single voxel away from edge.
        nudged_a = best_edges_df[['za', 'ya', 'xa']].values.astype(np.int32) - s
        nudged_b = best_edges_df[['zb', 'yb', 'xb']].values.astype(np.int32) + s

        nudged_a = np.maximum(nudged_a, (0,0,0))
        nudged_a = np.minimum(nudged_a, np.array(remapped_volume.shape) - 1)

        nudged_b = np.maximum(nudged_b, (0,0,0))
        nudged_b = np.minimum(nudged_b, np.array(remapped_volume.shape) - 1)

        # Keep only the nudged coordinates which retain the correct label;
        # revert the others to the original coordinate.
        keep_a = (remapped_volume[(*nudged_a.T,)] == best_edges_df['label_a'])
        keep_b = (remapped_volume[(*nudged_b.T,)] == best_edges_df['label_b'])

        best_edges_df.loc[keep_a, ['za', 'ya', 'xa']] = nudged_a[keep_a.values]
        best_edges_df.loc[keep_b, ['zb', 'yb', 'xb']] = nudged_b[keep_b.values]



    # Translate coordinates to global space
    best_edges_df.loc[:, ['za', 'ya', 'xa']] += brick.physical_box[0]
    best_edges_df.loc[:, ['zb', 'yb', 'xb']] += brick.physical_box[0]

    # Restore to original label set
    best_edges_df['label_a'] = reverse_mapper.apply(best_edges_df['label_a'].values)
    best_edges_df['label_b'] = reverse_mapper.apply(best_edges_df['label_b'].values)

    # Normalize
    swap_df_cols(best_edges_df, None, best_edges_df.eval('label_a > label_b'), ['a', 'b'])

    return best_edges_df


def _find_closest_approaches(volume, closest_scale, subset_groups):
    """
    Given a volume and one or more groups of labels,
    find intra-group "edges" (label pairs) for objects in the given volume that are
    close to one another, but don't actually touch.

    Note:
        The subset_groups argument must contain a 'group_cc' column,
        which identifies nodes for which an edge has already been found
        during the "direct adjacencies" step of this workflow.
        This allows us to avoid looking for edges that have already been found,
        and also avoid looking at edges that aren't necessary to join a node with
        the rest of its group.

        Instead, we focus on the closest approaches between nodes that span
        from one component in a group to another component in that group.

        For example, if a particular group has three components [1,2,3], [4,5], [6],
        then we will only look for closest approaches between the following pairs of nodes:
        [1,4], [1,5], [1,6], [2,4], [2,5], [2,6], [4,6], [5,6]
        (We won't examine [1,2], [1,3], [2,3], [4,5], since they are known to belong
        to the same group already.)

        This saves computation, but omits some close-but-not-touching edges between nodes
        in the same group, just because some other path consisting of direct adjacencies
        can be found between those two points.

    Args:
        volume:
            3D label volume, np.uint32

        closest_scale:
            If closest_scale > 0, then the "closest" points will be computed after
            downsampling the mask for each object at the given scale.
            (The returned point is still guaranteed to fall within the object at
            scale 0, but it may be a pixel or two away from the true point of
            closest approach.)

        subset_groups:
            DataFrame with columns [label, group, group_cc].
            Each grouped subset subset of labels is considered independently.
            Furthermore, we do not look for edges within the same group_cc,
            as explained above.

    Returns:
        DataFrame with columns:
            [label_a, label_b, za, ya, xa, zb, yb, xb, distance, edge_area]
        Note:
            ``edge_area`` will be 0 for all rows, since none of the body pairs
            physically touch in the volume (a precondition for the input).
    """
    assert volume.ndim == 3
    assert volume.dtype == np.uint32

    subset_groups = subset_groups[['label', 'group', 'group_cc']]

    # We can only process groups that contain at least two labels.
    # If no group contains at least two labels, we're done.
    if subset_groups['group'].value_counts().max() == 1:
        return None

    # We only find edges from one CC to another.
    # (We don't bother looking for edges within a pre-established CC)
    # Therefore, if a group contains only 1 CC, we don't deal with it.
    cc_counts = subset_groups.groupby('group')['group_cc'].agg('nunique')
    _kept_groups = cc_counts[cc_counts >= 2].index
    subset_groups = subset_groups.query('group in @_kept_groups')

    if len(subset_groups) == 0:
        return None
    
    def distanceTransformUint8(volume):
        # For the watershed below, the distance transform input need not be super-precise,
        # and vigra's watersheds() function operates MUCH faster on uint8 data.
        dt = vigra.filters.distanceTransform(volume)
        dt = (255*dt / dt.max()).astype(np.uint8)
        return dt

    def fill_gaps(volume):
        dt = distanceTransformUint8(volume)
        
        # The watersheds function annoyingly prints a bunch of useless messages to stdout,
        # so hide that stuff using this context manager.
        with stdout_redirected():
            ws, _max_label = vigra.analysis.watersheds(dt, seeds=volume, method='Turbo')
        return ws

    subset_edges = []
    for _group_id, group_df in subset_groups.groupby('group'):
        group_labels = pd.unique(group_df['label'])
        if len(group_labels) == 1:
            continue
        elif len(group_labels) <= 5:
            subset_edges.extend([*combinations(sorted(group_labels), 2)])
        else:
            # Rather than computing pairwise distances between all labels,
            # Figure out which labels are close to each other by filling the
            # gaps in the image and computing direct adjacencies.
            masked_vol = apply_mask_for_labels(volume, group_df['label'])
            filled_vol = fill_gaps(masked_vol)
            edges_df = compute_dense_rag_table(filled_vol)
            subset_edges.extend( edges_df[['label_a', 'label_b']].drop_duplicates().values.tolist() )

    subset_edges = pd.DataFrame(subset_edges, columns=['label_a', 'label_b'], dtype=np.uint64)

    subset_edges = subset_edges.merge(subset_groups, 'left',
                                      left_on='label_a', right_on='label').drop('label', axis=1)
    subset_edges = subset_edges.merge(subset_groups, 'left',
                                      left_on='label_b', right_on='label',  suffixes=['_a', '_b']).drop('label', axis=1)

    subset_edges = subset_edges.query('(group_a == group_b) and (group_cc_a != group_cc_b)')
    subset_edges = subset_edges[['label_a', 'label_b']].drop_duplicates()

    result_rows = []
    for (label_a, label_b) in subset_edges.values:
        coord_a, coord_b, distance = approximate_closest_approach(volume, label_a, label_b, closest_scale)

        # FIXME: Why is this check necessary? Both label_a and label_b should be present...
        if not np.isinf(distance):
            result_rows.append((label_a, label_b, *coord_a, *coord_b, distance))
    
    if len(result_rows) == 0:
        return None
    
    df = pd.DataFrame(result_rows, columns=['label_a', 'label_b', 'za', 'ya', 'xa', 'zb', 'yb', 'xb', 'distance'])

    # These objects don't touch, so their edge area is 0.
    # (Don't call this function for objects that do touch)
    df['edge_area'] = np.int32(0)

    touching_df = df.query('distance <= 1.0')
    if len(touching_df) > 0:
        path = 'unexpected-touching-objects-remapped.npy'
        np.save(path, touching_df)
        msg = f"I didn't expect you to call this function with objects that physically touch! See {path}"
        raise RuntimeError(msg)
    
    return df.astype({**EDGE_TABLE_TYPES, 'label_a': np.uint32, 'label_b': np.uint32})


def _find_and_select_central_edges(volume, subset_groups, subset_requirement):
    """
    Helper function.
    Comput the RAG of direct adjacencies among bodies in the given subset_groups,
    and select the most central edge location for each.
    
    The volume must already be of type np.uint32.
    The caller may need to apply a mapping the volume, bodies, and edges
    before calling this function.
    
    The coordinates in the returned DataFrame will be in terms of the
    local volume's coordinates (i.e. between (0,0,0) and volume.shape).
    
    TODO:
        Right now this function uses ilastikrag.Rag to find the adjacencies,
        because it's convenient.  But it does *slightly* more work than necessary,
        and it would be easy to copy the functions we need into this code base.
        Plus, the conversion to uint32 is probably unnecessary...
        
    FIXME: subset_requirement is not respected.
    """
    edges_df = compute_dense_rag_table(volume)
    subset_groups = subset_groups[['label', 'group']]
    
    # Keep only the edges that belong in the same group(s)
    edges_df = edges_df.merge(subset_groups, 'inner', left_on='label_a', right_on='label').drop('label', axis=1)
    edges_df = edges_df.merge(subset_groups, 'inner', left_on='label_b', right_on='label',
                                        suffixes=['_a', '_b']).drop('label', axis=1)
    edges_df = edges_df.query('group_a == group_b')
    edges_df = edges_df.drop(['group_a', 'group_b'], axis=1)
    
    # Some coordinates may be listed twice for a given edge pair, since the
    # same coordinate might be "above" and "to the left" of the partner
    # object if the edge boundary "jagged".
    # Subjectively, it's better not to double-count such edges when computing
    # the centroid of the edge's coordinates.
    # Also, it's possible that two groups include the same label pair,
    # but we still only return it once.
    edges_df.drop_duplicates(['label_a', 'label_b', 'z', 'y', 'x'], inplace=True)

    if len(edges_df) == 0:
        return None # No edges left after filtering

    edges_df['distance'] = np.float32(1.0)

    # Find most-central edge in each group    
    best_edges_df = select_central_edges(edges_df, ['z', 'y', 'x'])

    # Compute the 'right-hand' coordinates
    best_edges_df.rename(columns={'z': 'za', 'y': 'ya', 'x': 'xa'}, inplace=True)
    best_edges_df['zb'] = best_edges_df['za']
    best_edges_df['yb'] = best_edges_df['ya']
    best_edges_df['xb'] = best_edges_df['xa']

    z_edges = (best_edges_df['axis'] == 'z')
    y_edges = (best_edges_df['axis'] == 'y')
    x_edges = (best_edges_df['axis'] == 'x')

    best_edges_df.loc[z_edges, 'zb'] += 1
    best_edges_df.loc[y_edges, 'yb'] += 1
    best_edges_df.loc[x_edges, 'xb'] += 1

    swap_df_cols(best_edges_df, ['z', 'y', 'x'], ~(best_edges_df['forwardness']), ['a', 'b'])
    del best_edges_df['forwardness']
    del best_edges_df['axis']

    return best_edges_df[['label_a', 'label_b', 'za', 'ya', 'xa', 'zb', 'yb', 'xb', 'distance', 'edge_area']]


def compute_dense_rag_table(volume):
    """
    Find all voxel-level adjacencies in the volume,
    except for adjacencies to ID 0.
    
    Returns:
        dataframe with columns:
        ['label_a', 'label_b', 'forwardness', 'z', 'y', 'x', 'axis']
        where ``label_a < label_b`` for all rows, and `forwardness` indicates whether
        label_a is on the 'left' (upper) or on the 'right' (lower) side
        of the voxel boundary.
    """
    assert volume.dtype == np.uint32
    rag = Rag(vigra.taggedView(volume, 'zyx'))
    
    # Edges are stored by axis -- concatenate them all.
    edges_z, edges_y, edges_x = rag.dense_edge_tables.values()

    del rag

    if len(edges_z) == len(edges_y) == len(edges_x) == 0:
        return None # No edges detected
    
    edges_z['axis'] = 'z'
    edges_y['axis'] = 'y'
    edges_x['axis'] = 'x'
    
    all_edges = list(filter(len, [edges_z, edges_y, edges_x]))
    edges_df = pd.concat(all_edges, ignore_index=True)
    edges_df.rename(columns={'sp1': 'label_a', 'sp2': 'label_b'}, inplace=True)
    del edges_df['edge_label']

    # Filter: not interested in label 0
    edges_df.query("label_a != 0 and label_b != 0", inplace=True)

    return edges_df

def select_central_edges(all_edges_df, coord_cols=['za', 'ya', 'xa']):
    """
    Given a DataFrame with at least columns [label_a, label_b, *coord_cols, distance],
    select the row with the most-central point for each
    unique [label_a, label_b] pair.

    The most-central point is defined as the row whose coordinates are
    closest to the centroid of the points that belong to the group of
    rows with matching [label_a, label_b] columns.
    
    If a column named 'edge_area' is present, it specifies the weight that
    should be used when computing the centroid of the edge points.
    
    Any extra columns are passed on in the output.
    """
    all_edges_df = all_edges_df.copy()
    if 'edge_area' not in all_edges_df:
        all_edges_df['edge_area'] = np.int32(1)

    orig_columns = all_edges_df.columns

    ##
    ## Compute (weighted) centroids
    ##
    Z, Y, X = coord_cols
    
    all_edges_df['cz'] = all_edges_df.eval(f'{Z} * edge_area')
    all_edges_df['cy'] = all_edges_df.eval(f'{Y} * edge_area')
    all_edges_df['cx'] = all_edges_df.eval(f'{X} * edge_area')

    centroids_df = ( all_edges_df[['label_a', 'label_b', 'cz', 'cy', 'cx', 'edge_area']]
                       .groupby(['label_a', 'label_b'], sort=False).sum() )

    centroids_df['cz'] = centroids_df.eval('cz / edge_area')
    centroids_df['cy'] = centroids_df.eval('cy / edge_area')
    centroids_df['cx'] = centroids_df.eval('cx / edge_area')

    # Clean up: Drop the columns we added for centroid computation
    # (don't let these mess up the merge below)
    all_edges_df.drop(['cz', 'cy', 'cx', 'edge_area'], axis=1, inplace=True)
    
    # Distribute centroids back to the full table
    all_edges_df = all_edges_df.merge(centroids_df, on=['label_a', 'label_b'], how='left')
    
    # Calculate each distance to the centroid
    all_edges_df['distance_to_centroid'] = all_edges_df.eval(f'sqrt( ({Z}-cz)**2 + ({Y}-cy)**2 + ({X}-cx)**2 )')

    # Find the best row for each edge (min distance)
    min_distance_df = all_edges_df[['label_a', 'label_b', 'distance_to_centroid']].groupby(['label_a', 'label_b'], sort=False).idxmin()
    min_distance_df.rename(columns={'distance_to_centroid': 'best_row'}, inplace=True)

    # Select the best rows from the original data and return
    best_edges_df = all_edges_df.loc[min_distance_df['best_row'], orig_columns].copy()
    return best_edges_df


def select_closest_edges(all_edges_df):
    """
    Filter the given edges to include only the edge with
    the minimum distance for each [label_a,label_b] pair.
    """
    return (all_edges_df
                .sort_values(['label_a', 'label_b', 'distance'])
                .drop_duplicates(['label_a', 'label_b'], keep='first'))
