import os
import copy
import logging

import vigra
import numpy as np
import pandas as pd

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, swap_df_cols, closest_approach, SparseBlockMask, connected_components_nonconsecutive

from ilastikrag import Rag
from dvidutils import LabelMapper

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
    'edge_area': np.int32 }

EDGE_TABLE_COLS = list(EDGE_TABLE_TYPES.keys())


class FindAdjacencies(Workflow):
    """
    Workflow to find all adjacent bodies in a segmentation volume,
    and output a coordinate for each that lies along the adjacency
    boundary, and somewhat near the centroid of the boundary.
    
    Note: This workflow performs well on medium-sized volumes,
          or on large volumes when filtering the set of edges using either
          the restrict-bodies or restrict-edges settings in the config.
          It will not perform well on an unfiltered hemibrain-sized volume,
          which would result in billions of edges.
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
            "subset-labels": BodyListSchema,
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
            "find-closest": {
                "description": "For body pairs that do not physically touch, find the points at which they come closest to eachother.\n"
                               "Only permitted when using a subset option.\n",
                "type": "boolean",
                "default": False
            },
            "compute-cc": {
                "description": "Once the edges have been found, run connected components on the\n"
                               "resulting graph, and export the component ID in the results.\n",
                "type": "boolean",
                "default": True
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
    
    FindAdjacenciesOptionsSchema["properties"]["subset-labels"]["description"] += (
        "If provided, this list will be used to limit the set adjacencies returned.\n"
        "See 'subset-labels-requirement'\n")

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
        find_closest = options["find-closest"]

        num_subsets = sum([ bool(options["subset-labels"]),
                            bool(options["subset-edges"]),
                            bool(options["subset-label-groups"]) ])
        if num_subsets > 1:
            raise RuntimeError("You cannot specify more than one subset mechanism. "
                               "Provide either subset-labels, subset-edges, or subset-label-groups.")


        if find_closest and subset_requirement != 2:
            raise RuntimeError("Can't use find-closest unless subset-requirement == 2")

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
        find_closest = options["find-closest"]

        self.resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])
        volume_service = VolumeService.create_from_config(input_config, self.resource_mgr_client)

        subset_groups = self._load_label_groups(volume_service)

        brickwall = self.init_brickwall(volume_service, subset_groups)

        with Timer("Finding direct adjacencies", logger):
            def find_adj(brick):
                # FIXME: Instead of broadcasting the entire brick_coords_df to all tasks,
                #        it might be better to distribute each brick's rows.
                #        (But that will involve a dask join step...)
                #        The brick_coords_df should generally be under 1 GB, anyway...
                return find_edges_in_brick(brick, 'adjacent', subset_groups, subset_requirement)
            adjacent_edge_tables = brickwall.bricks.map(find_adj).compute()

        with Timer("Combining/filtering direct adjacencies", logger):
            adjacent_edge_tables = list(filter(lambda t: t is not None, adjacent_edge_tables))
            if len(adjacent_edge_tables) == 0:
                all_adjacent_edges_df = pd.DataFrame([], columns=EDGE_TABLE_COLS).astype(EDGE_TABLE_TYPES)
                best_adjacent_edges_df = all_adjacent_edges_df
            else:
                all_adjacent_edges_df = pd.concat(adjacent_edge_tables, ignore_index=True)
                best_adjacent_edges_df = select_central_edges(all_adjacent_edges_df, ['za', 'ya', 'xa'])

        # No non-adjacencies by default
        all_nonadjacent_edges_df = pd.DataFrame([], columns=EDGE_TABLE_COLS).astype(EDGE_TABLE_TYPES)
        best_nonadjacent_edges_df = all_nonadjacent_edges_df

        if find_closest:
            np.save('all-adjacent-brick-edges-for-debug.npy', all_adjacent_edges_df.to_records(index=False))
            np.save('best-adjacent-brick-edges-for-debug.npy', best_nonadjacent_edges_df.to_records(index=False))
            
            with Timer("Finding closest approaches", logger):
                ignored_pairs = best_adjacent_edges_df[['label_a', 'label_b']]
                def find_closest(brick):
                    return find_edges_in_brick(brick, 'closest', subset_groups, subset_requirement, ignored_pairs)
                nonadjacent_edge_tables = brickwall.bricks.map(find_closest).compute()

            with Timer("Combining closest approaches", logger):
                nonadjacent_edge_tables = list(filter(lambda t: t is not None, nonadjacent_edge_tables))
                if len(nonadjacent_edge_tables) != 0:
                    all_nonadjacent_edges_df = pd.concat(nonadjacent_edge_tables, ignore_index=True)
                    best_nonadjacent_edges_df = select_closest_edges(all_nonadjacent_edges_df)

        with Timer("Sorting edges", logger):
            final_edges_df = pd.concat((best_adjacent_edges_df, best_nonadjacent_edges_df))
            # This sort isn't strictly necessary, but makes it
            # easier to diff results of one execution vs another.
            final_edges_df.sort_values(list(final_edges_df.columns), inplace=True)

        if options["compute-cc"]:
            max_distance = options["cc-distance-threshold"]
            nodes = pd.unique(final_edges_df[['label_a', 'label_b']].values.reshape(-1))
            
            thresholded_edges_df = final_edges_df.query('distance <= @max_distance')
            edges = thresholded_edges_df[['label_a', 'label_b']].values
            
            cc = 1 + connected_components_nonconsecutive(edges, nodes)
            cc_df = pd.DataFrame({ 'label_a': nodes, 'cc': cc.astype(np.int32) })
            final_edges_df = final_edges_df.merge(cc_df, 'left', 'label_a')
            final_edges_df['cc'] = final_edges_df['cc'].astype(np.int32)

            # Edges that were not used might be part of two different components.
            final_edges_df.loc[final_edges_df['distance'] > max_distance, 'cc'] = np.int32(-1)

        with Timer("Appending groups"):
            final_edges_df = final_edges_df.merge(subset_groups, 'left', left_on='label_a', right_on='label').drop('label', axis=1)
            final_edges_df = final_edges_df.merge(subset_groups, 'left', left_on='label_b', right_on='label',
                                                  suffixes=['_a', '_b']).drop('label', axis=1)
            final_edges_df = final_edges_df.query('group_a == group_b')
            final_edges_df = final_edges_df.rename(columns={'group_a': 'group'}).drop('group_b', axis=1)
            
            # Put group first, and sort by group
            cols = final_edges_df.columns.tolist()
            cols.remove('group')
            cols.insert(0, 'group')
            final_edges_df = final_edges_df[cols]

        with Timer("Writing edges", logger):
            final_edges_df.to_csv(options["output-table"], header=True, index=False)

            # Save in numpy format, too.
            npy_path = options["output-table"][:-4] + '.npy'
            np.save(npy_path, final_edges_df.to_records(index=False))

            # For debugging and analysis, write complete table of edges we found in all bricks,
            # even though some pairs may appear more than once.
            all_edges_df = pd.concat((best_adjacent_edges_df, best_nonadjacent_edges_df))
            np.save('all-unfiltered-brick-edges-for-debug.npy', all_edges_df.to_records(index=False))


    def init_brickwall(self, volume_service, subset_groups):
        try:
            brick_coords_df = volume_service.sparse_brick_coords_for_label_groups(subset_groups)
            brick_shape = volume_service.preferred_message_shape
            coords = brick_coords_df[['z', 'y', 'x']].values // brick_shape
            sbm = SparseBlockMask.create_from_lowres_coords(coords, brick_shape)
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
            brickwall = BrickWall.from_volume_service(volume_service, 0, None, self.client, target_partition_size_voxels, halo, sbm)

        return brickwall


def find_edges_in_brick(brick, method='adjacent', subset_groups=[], subset_requirement=2, ignored_pairs=[]):
    """
    Find all pairs of adjacent labels in the given brick,
    and find the central-most point along the edge between them.
    
    (Edges to/from label 0 are discarded.)
    
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
    ignored_pairs = np.asarray(ignored_pairs, dtype=np.uint64).reshape((-1,2))
    
    # ilastikrag requires uint32, so remap to consecutive ints
    brick_labels = set(pd.unique(brick.volume.reshape(-1)))
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
    remapped_subset_groups = subset_groups.copy()
    remapped_subset_groups['label'] = mapper.apply(subset_groups['label'].values)

    if method == 'adjacent':
        best_edges_df = _find_and_select_central_edges(remapped_volume, remapped_subset_groups, subset_requirement)
    elif method == 'closest':
        remapped_ignored_pairs = mapper.apply_with_default(ignored_pairs, 0)
        remapped_ignored_edges = pd.DataFrame(remapped_ignored_pairs, columns=['label_a', 'label_b'], dtype=np.uint32)
        remapped_ignored_edges = remapped_ignored_edges.query('label_a != 0 and label_b != 0')
        best_edges_df = _find_closest_approaches(remapped_volume, remapped_subset_groups, remapped_ignored_edges)
    else:
        raise AssertionError(f"Unknown method: {method}")

    if best_edges_df is None:
        return None

    # Translate coordinates to global space
    best_edges_df.loc[:, ['za', 'ya', 'xa']] += brick.physical_box[0]
    best_edges_df.loc[:, ['zb', 'yb', 'xb']] += brick.physical_box[0]

    # Restore to original label set
    best_edges_df['label_a'] = reverse_mapper.apply(best_edges_df['label_a'].values)
    best_edges_df['label_b'] = reverse_mapper.apply(best_edges_df['label_b'].values)

    # Normalize
    swap_df_cols(best_edges_df, None, best_edges_df.eval('label_a > label_b'), ['a', 'b'])
    
    return best_edges_df


def _find_closest_approaches(volume, subset_groups, ignored_edges):
    """
    Given a volume and a list of list of edges (body pairs),
    Find the points at which the two bodies come closest to one
    another in the volume.
    
    Note:
        None of the body pairs of interest should actually touch each
        other in the volume.  Use ``ignored_edges`` to explicitly avoid
        analyzing bodies that actually touch.
    
    Args:
        volume:
            3D label volume, np.uint32
            
        subset_bodies:
            Consider all pairwise combinations of bodies in this list
            and use it instead of subset_edges.
            Cannot be used with subset_edges.
        
        subset_edges:
            The list of body pairs to analyze.  Cannot be used with subset_bodies.
        
        ignored_edges:
            Explicitly ignore any edges (body pairs) in this list.
    
    Returns:
        DataFrame with columns:
            [label_a, label_b, za, ya, xa, zb, yb, xb, distance, edge_area]
        Note:
            ``edge_area`` will be 0 for all rows, since none of the body pairs
            physically touch in the volume (a precondition for the input).
    """
    assert volume.ndim == 3
    assert volume.dtype == np.uint32

    if subset_groups['group'].value_counts().max() == 1:
        return None

    # Normalize
    ignored_edges = ignored_edges.copy()
    swap_df_cols(ignored_edges, None, ignored_edges.eval('label_a > label_b') )

    # Compute all intra-group pair combinations
    subset_edges = subset_groups.merge(subset_groups, 'inner', 'group', suffixes=['_a', '_b'])
    swap_df_cols(subset_edges, None, subset_edges.eval('label_a > label_b') ) # Normalize
    subset_edges = subset_edges.drop_duplicates(['label_a', 'label_b']).query('label_a != label_b')

    # Exclude any pre-specified ignored_edges
    subset_edges = subset_edges.merge(ignored_edges, 'left', ['label_a', 'label_b'], indicator='source')
    subset_edges = subset_edges.query('source == "left_only"')
    subset_edges = subset_edges.drop('source', axis=1)

    result_rows = []
    for (label_a, label_b) in subset_edges[['label_a', 'label_b']].values:
        coord_a, coord_b, distance = closest_approach(volume, label_a, label_b, check_present=False)
        result_rows.append((label_a, label_b, *coord_a, *coord_b, distance))
    
    if len(result_rows) == 0:
        return None
    
    df = pd.DataFrame(result_rows, columns=['label_a', 'label_b', 'za', 'ya', 'xa', 'zb', 'yb', 'xb', 'distance'])

    # These objects don't touch, so their edge area is 0.
    # (Don't call this function for objects that do touch)
    df['edge_area'] = np.int32(0)

    assert not (df['distance'] <= 1.0).any(), \
        "I didn't expect you to call this function with objects that physically touch!"
    
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
    min_selections = all_edges_df.groupby(['label_a', 'label_b'], sort=False).agg({'distance': 'idxmin'})
    return all_edges_df.loc[min_selections['distance'].values]
