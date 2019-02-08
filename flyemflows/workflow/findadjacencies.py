import os
import copy
import logging
from itertools import combinations

import vigra
import numpy as np
import pandas as pd

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, Grid, swap_df_cols, closest_approach

from ilastikrag import Rag
from dvidutils import LabelMapper

from ..brick import BrickWall
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService
from .util.config_helpers import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)


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
            "use-halo": {
                "description": "Whether or not to construct overlapping bricks before checking for adjacencies,\n"
                               "to catch possible inter-block adjacencies that might have been missed if no halo were used.\n"
                               "(This incurs a significant performance penalty, introduces an extra job boundary (with associated RAM overhead),\n"
                               "and is unlikely to change the results much.)\n",
                "type": "boolean",
                "default": False
            },
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
                               "Only permitted when using subset-labels or subset-edges.\n",
                "type": "boolean",
                "default": False
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
        if options["subset-labels"] and options["subset-edges"]:
            raise RuntimeError("Can't use both subset-labels and subset-edges")


    def execute(self):
        self._sanitize_config()
        input_config = self.config["input"]
        options = self.config["findadjacencies"]
        resource_config = self.config["resource-manager"]

        resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])
        volume_service = VolumeService.create_from_config(input_config, os.getcwd(), resource_mgr_client)

        is_supervoxels = False
        if isinstance(volume_service.base_service, DvidVolumeService):
            is_supervoxels = volume_service.base_service.supervoxels

        subset_requirement = options["subset-labels-requirement"]

        # Load body list and eliminate duplicates
        subset_bodies = load_body_list(options["subset-labels"], is_supervoxels)
        subset_bodies = set(subset_bodies)
        if len(subset_bodies) == 1 and subset_requirement == 2:
            raise RuntimeError("Only one body was listed in subset-bodies.  No edges would be found!")
        
        subset_edges = np.zeros((0,2), np.uint64)
        if options["subset-edges"]:
            subset_edges = pd.read_csv(options["subset-edges"], dtype=np.uint64, header=0, names=['label_a', 'label_b']).values
            subset_edges.sort(axis=1)
        subset_edges = pd.DataFrame(subset_edges, columns=['label_a', 'label_b'], dtype=np.uint64)

        # Remove invalid edges and eliminate duplicates
        subset_edges = subset_edges.query('label_a != label_b').drop_duplicates(['label_a', 'label_b'])

        find_closest = options["find-closest"]
        if find_closest and (len(subset_bodies) == 0 and len(subset_edges) == 0):
            raise RuntimeError("Can't use find-closest unless you provide a list of subset-edges or subset-bodies.")
        
        if find_closest and subset_requirement != 2:
            raise RuntimeError("Can't use find-closest unless subset-requirement == 2")

        with Timer("Initializing BrickWall", logger):
            # Aim for 2 GB RDD partitions when loading segmentation
            GB = 2**30
            target_partition_size_voxels = 2 * GB // np.uint64().nbytes
            brickwall = BrickWall.from_volume_service(volume_service, 0, None, self.client, target_partition_size_voxels)

            if options["use-halo"]:
                overlapping_grid = Grid(brickwall.grid.block_shape, halo=1)
                brickwall = brickwall.realign_to_new_grid(overlapping_grid)

        with Timer("Finding adjacencies in bricks", logger):
            def find_adj(brick):
                return find_adjacencies_in_brick(brick, subset_bodies, subset_requirement, subset_edges, find_closest)
            
            brick_edge_tables = brickwall.bricks.map(find_adj).compute()
            brick_edge_tables = list(filter(lambda t: t is not None, brick_edge_tables))
            if not brick_edge_tables:
                raise RuntimeError("No edges were found.")

        with Timer("Combining brick results", logger):
            all_edges_df = pd.concat(brick_edge_tables, ignore_index=True)
            
            adjacent_edges_df = all_edges_df.query('distance == 1.0')
            best_adj_edges_df = select_central_edges(adjacent_edges_df, ['za', 'ya', 'xa'])

            if not find_closest:
                best_edges_df = best_adj_edges_df
            else:
                # Find the non-adjacent edges that had no adjacent edge
                nonadjacent_edges_df = all_edges_df.merge(best_adj_edges_df[['label_a', 'label_b']],
                                                          'left', ['label_a', 'label_b'],
                                                          indicator='source').query("source == 'left_only'")
                del nonadjacent_edges_df['source']
                
                best_nonadj_edges_df = select_closest_edges(nonadjacent_edges_df)
                best_edges_df = pd.concat((best_adj_edges_df, best_nonadj_edges_df))

        with Timer("Writing edges", logger):
            best_edges_df.to_csv(options["output-table"], header=True, index=False)


def find_adjacencies_in_brick(brick, subset_bodies=[], subset_requirement=1, subset_edges=[], find_closest=False):
    """
    Find all pairs of adjacent labels in the given brick,
    and find the central-most point along the edge between them.
    
    (Edges to/from label 0 are discarded.)
    
    Returns:
        If the brick contains no edges at all (other than edges to label 0), return None.
        
        Otherwise, returns pd.DataFrame with columns:
            [label_a, label_b, forwardness, z, y, x, axis, edge_area, distance].
        
        where label_a < label_b,
        'axis' indicates which axis the edge crosses at the chosen coordinate,
        
        (z,y,x) is always given as the coordinate to the left/above/front of the edge
        (depending on the axis).
        
        If 'forwardness' is True, then the given coordinate falls on label_a and
        label_b is one voxel "after" it (to the right/below/behind the coordinate).
        Otherwise, the coordinate falls on label_b, and label_a is "after".
        
        And 'edge_area' is the total count of the voxels touching both labels.
    """
    # ilastikrag requires uint32, so remap to consecutive ints
    brick_labels = np.sort(pd.unique(brick.volume.flat))
    consecutive_labels = np.arange(1, len(brick_labels)+1, dtype=np.uint32)

    # Preserve label 0
    if brick_labels[0] == 0:
        consecutive_labels[0] = 0

    if len(brick_labels) == 1:
        return None # brick is solid -- no edges
    
    mapper = LabelMapper(brick_labels, consecutive_labels)
    reverse_mapper = LabelMapper(consecutive_labels, brick_labels)

    remapped_subset_bodies = None
    if len(subset_bodies) > 0:
        subset_bodies = set(brick_labels).intersection(subset_bodies)
        if not subset_bodies:
            return None # None of the subset bodies are present in this brick
        subset_bodies = np.fromiter(subset_bodies, dtype=np.uint64)
        remapped_subset_bodies = set(mapper.apply(subset_bodies))

    remapped_subset_edges = None
    if len(subset_edges) > 0:
        # Discard edges that can't be found in this brick because one or both labels aren't present.
        subset_edges = subset_edges.query('label_a in @brick_labels and label_b in @brick_labels')
        if len(subset_edges) == 0:
            return None # None of the subset edges are present in this brick
        remapped_subset_edges = pd.DataFrame(mapper.apply(subset_edges.values), columns=['label_a', 'label_b'])

    # Construct RAG -- finds all edges in the volume, on a per-pixel basis.
    remapped_brick = mapper.apply(brick.volume)
    best_edges_df = _find_best_edges(remapped_brick, remapped_subset_bodies, subset_requirement, remapped_subset_edges)

    if find_closest:
        assert subset_requirement == 2, \
            "Can't use 'find_closest' mode unless subset_requirement == 2"

        ignored_edges = None
        if best_edges_df is not None:
            ignored_edges = best_edges_df[['label_a', 'label_b']]
        
        close_edges_df = _find_closest_approaches(remapped_brick, remapped_subset_bodies, remapped_subset_edges, ignored_edges)
        if close_edges_df is not None:
            if best_edges_df is None:
                best_edges_df = close_edges_df
            else:
                assert set(best_edges_df.columns) == set(close_edges_df.columns)
                best_edges_df = pd.concat((best_edges_df, close_edges_df))

    if best_edges_df is None:
        return None

    # Translate coordinates to global space
    best_edges_df.loc[:, ['za', 'ya', 'xa']] += brick.physical_box[0]
    best_edges_df.loc[:, ['zb', 'yb', 'xb']] += brick.physical_box[0]

    # Restore to original label set
    best_edges_df['label_a'] = reverse_mapper.apply(best_edges_df['label_a'].values)
    best_edges_df['label_b'] = reverse_mapper.apply(best_edges_df['label_b'].values)

    swap_df_cols(best_edges_df, None, best_edges_df.eval('label_a > label_b'), ['a', 'b'])
    
    return best_edges_df


def _find_closest_approaches(volume, subset_bodies, subset_edges, ignored_edges):
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
    assert (subset_bodies is None) or (subset_edges is None), \
        "You can't use subset-bodies and subset-edges.  Pick one or the other."
    assert (subset_bodies is not None) or (subset_edges is not None), \
        "You can't use find-closest-approaches unless you are using subset-edges or subset-bodies."

    if subset_bodies and len(subset_bodies) == 1:
        return None
    
    if subset_bodies is not None:
        assert isinstance(subset_bodies, set)
        subset_edges = list(combinations(subset_bodies, 2))
        subset_edges = np.array(subset_edges)
        subset_edges.sort(axis=1)
        subset_edges = pd.DataFrame(subset_edges, columns=['label_a', 'label_b'])

    if ignored_edges is not None:
        # Normalize
        ignored_edges = ignored_edges.copy()
        swap_df_cols(ignored_edges, None, ignored_edges.eval('label_a > label_b') )
        swap_df_cols(subset_edges, None, subset_edges.eval('label_a > label_b') )
    
        # Discard edges that should be ignored (i.e. they were already found by other means)
        subset_edges = subset_edges.merge(ignored_edges, 'left', ['label_a', 'label_b'],
                                          indicator='source').query("source == 'left_only'")
        subset_edges.drop(columns=['source'], inplace=True)
    
    approach_rows = []
    for row in subset_edges.itertuples(index=False):
        coord_a, coord_b, distance = closest_approach(volume, row.label_a, row.label_b, check_present=False)
        approach_rows.append((row.label_a, row.label_b, *coord_a, *coord_b, distance))
    
    if len(approach_rows) == 0:
        return None
    
    df = pd.DataFrame(approach_rows, columns=['label_a', 'label_b', 'za', 'ya', 'xa', 'zb', 'yb', 'xb', 'distance'])
    df['label_a'] = df['label_a'].astype(np.uint32)
    df['label_b'] = df['label_b'].astype(np.uint32)
    df['za'] = df['za'].astype(np.int32)
    df['ya'] = df['ya'].astype(np.int32)
    df['xa'] = df['xa'].astype(np.int32)
    df['zb'] = df['zb'].astype(np.int32)
    df['yb'] = df['yb'].astype(np.int32)
    df['xb'] = df['xb'].astype(np.int32)
    df['distance'] = df['distance'].astype(np.float32)
    
    assert not (df['distance'] <= 1.0).any(), \
        "I didn't expect you to call this function with objects that physically touch!"
    
    # These objects don't touch
    # (Don't call this function for objects that do touch)
    df['edge_area'] = np.int32(0)
    
    return df[['label_a', 'label_b', 'za', 'ya', 'xa', 'zb', 'yb', 'xb', 'distance', 'edge_area']]


def _find_best_edges(volume, subset_bodies, subset_requirement, subset_edges):
    """
    Helper function.
    Find the "best" (most central) edges in a volume.
    
    The volume must already be of type np.uint32.
    The caller may need to apply a mapping the volume, bodies, and edges
    before calling this function.
    
    The coordinates in the returned DataFrame will be in terms of the
    local volume's coordinates (i.e. between (0,0,0) and volume.shape).
    """
    assert volume.dtype == np.uint32
    rag = Rag(vigra.taggedView(volume, 'zyx'))
    
    # Edges are stored by axis -- concatenate them all.
    edges_z, edges_y, edges_x = rag.dense_edge_tables.values()

    if len(edges_z) == len(edges_y) == len(edges_x) == 0:
        return None # No edges detected
    
    edges_z['axis'] = 'z'
    edges_y['axis'] = 'y'
    edges_x['axis'] = 'x'
    
    all_edges = list(filter(len, [edges_z, edges_y, edges_x]))
    all_edges_df = pd.concat(all_edges, ignore_index=True)
    all_edges_df.rename(columns={'sp1': 'label_a', 'sp2': 'label_b'}, inplace=True)
    del all_edges_df['edge_label']

    # Some coordinates may be listed twice for a given edge pair, since the
    # same coordinate might be "above" and "to the left" of the partner
    # object if the edge boundary "jagged".
    # Subjectively, it's better not to double-count such edges when computing
    # the centroid of the edge's coordinates.
    all_edges_df.drop_duplicates(['label_a', 'label_b', 'z', 'y', 'x'], inplace=True)

    # Filter: not interested in label 0
    all_edges_df.query("label_a != 0 and label_b != 0", inplace=True)

    # Filter by subset-bodies
    if subset_bodies is not None:
        if subset_requirement == 1:
            all_edges_df.query("label_a in @subset_bodies or label_b in @subset_bodies", inplace=True)
        if subset_requirement == 2:
            all_edges_df.query("label_a in @subset_bodies and label_b in @subset_bodies", inplace=True)

    # Filter by subset-edges
    if subset_edges is not None:
        all_edges_df = all_edges_df.merge(subset_edges, on=['label_a', 'label_b'], how='inner')
    
    if len(all_edges_df) == 0:
        return None # No edges left after filtering

    all_edges_df['distance'] = np.float32(1.0)

    # Find most-central edge in each group    
    best_edges_df = select_central_edges(all_edges_df, ['z', 'y', 'x'])

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
                       .groupby(['label_a', 'label_b']).sum() )

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
    min_distance_df = all_edges_df[['label_a', 'label_b', 'distance_to_centroid']].groupby(['label_a', 'label_b']).idxmin()
    min_distance_df.rename(columns={'distance_to_centroid': 'best_row'}, inplace=True)

    # Select the best rows from the original data and return
    best_edges_df = all_edges_df.loc[min_distance_df['best_row'], orig_columns].copy()
    return best_edges_df


def select_closest_edges(all_edges_df):
    """
    Filter the given edges to include only the edge with
    the minimum distance for each [label_a,label_b] pair.
    """
    min_selections = all_edges_df.groupby(['label_a', 'label_b']).agg({'distance': 'idxmin'})
    return all_edges_df.loc[min_selections['distance'].values]
