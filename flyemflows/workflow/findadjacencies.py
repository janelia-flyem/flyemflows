import os
import copy
import logging

import vigra
import numpy as np
import pandas as pd

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, Grid

from ilastikrag import Rag
from dvidutils import LabelMapper

from ..brick import BrickWall
from ..volumes import VolumeService, SegmentationVolumeSchema
from . import Workflow

logger = logging.getLogger(__name__)


class FindAdjacencies(Workflow):
    """
    Workflow to find all adjacent bodies in a segmentation volume,
    and output a coordinate for each that lies along the adjacency
    boundary, and somewhat near the centroid of the boundary.
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
                               "and is unlikely to change the results much.)",
                "type": "boolean",
                "default": False
            },
            "output-table": {
                "description": "Results file.  Must be .csv for now, and must contain at least columns x,y,z",
                "type": "string",
                "default": "point-samples.csv"
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


    def execute(self):
        input_config = self.config["input"]
        options = self.config["findadjacencies"]
        resource_config = self.config["resource-manager"]

        resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])
        volume_service = VolumeService.create_from_config(input_config, os.getcwd(), resource_mgr_client)

        with Timer("Initializing BrickWall", logger):
            # Aim for 2 GB RDD partitions when loading segmentation
            GB = 2**30
            target_partition_size_voxels = 2 * GB // np.uint64().nbytes
            brickwall = BrickWall.from_volume_service(volume_service, 0, None, self.client, target_partition_size_voxels)

            if options["use-halo"]:
                overlapping_grid = Grid(brickwall.grid.block_shape, halo=1)
                brickwall = brickwall.realign_to_new_grid(overlapping_grid)

        with Timer("Finding adjacencies in bricks", logger):        
            brick_edge_tables = brickwall.bricks.map(find_adjacencies_in_brick).compute()
            brick_edge_tables = list(filter(lambda t: t is not None, brick_edge_tables))

        with Timer("Combining brick results", logger):        
            all_edges_df = pd.concat(brick_edge_tables, ignore_index=True)
            best_edges_df = select_central_edges(all_edges_df)

        with Timer("Writing edges", logger):
            best_edges_df.to_csv(options["output-table"], header=True, index=False)


def find_adjacencies_in_brick(brick):
    """
    Find all pairs of adjacent labels in the given brick,
    and find the central-most point along the edge between them.
    
    (Edges to/from label 0 are discarded.)
    
    Returns:
        If the brick contains no edges at all (other than edges to label 0), return None.
        
        Otherwise, returns pd.DataFrame with columns:
            [label_a, label_b, forwardness, z, y, x, axis, edge_area].
        
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

    if brick_labels[0] == 0:
        consecutive_labels[0] = 0

    if len(brick_labels) == 1:
        return None
    
    mapper = LabelMapper(brick_labels, consecutive_labels)
    reverse_mapper = LabelMapper(consecutive_labels, brick_labels)
    
    remapped_brick = mapper.apply(brick.volume)

    # Construct RAG -- finds all edges in the volume, on a per-pixel basis.
    rag = Rag(vigra.taggedView(remapped_brick, 'zyx'))
    
    # Edges are stored by axis -- concatenate them all.
    edges_z, edges_y, edges_x = rag.dense_edge_tables.values()

    if len(edges_z) == len(edges_y) == len(edges_x) == 0:
        return None
    
    edges_z['axis'] = 'z'
    edges_y['axis'] = 'y'
    edges_x['axis'] = 'x'
    
    all_edges_df = pd.concat([edges_z, edges_y, edges_x], ignore_index=True)
    all_edges_df.rename(columns={'sp1': 'label_a', 'sp2': 'label_b'}, inplace=True)
    all_edges_df.query("label_a != 0 and label_b != 0", inplace=True)
    del all_edges_df['edge_label']

    # Some coordinates may be listed twice for a given edge pair, since the
    # same coordinate might be "above" and "to the left" of the partner
    # object if the edge boundary "jagged".
    # Subjectively, it's better not to double-count such edges when computing
    # the centroid of the edge's coordinates.
    all_edges_df.drop_duplicates(['label_a', 'label_b', 'z', 'y', 'x'], inplace=True)
    
    best_edges_df = select_central_edges(all_edges_df)
    best_edges_df.loc[:, ['z', 'y', 'x']] += brick.physical_box[0]

    best_edges_df['label_a'] = reverse_mapper.apply(best_edges_df['label_a'].values)
    best_edges_df['label_b'] = reverse_mapper.apply(best_edges_df['label_b'].values)    
    return best_edges_df


def select_central_edges(all_edges_df):
    """
    Given a DataFrame with at least columns [label_a, label_b, z, y, x],
    select the row with the most-central point for each
    unique [label_a, label_b] pair.

    The most-central point is defined as the row whose coordinates are
    closest to the centroid of the points that belong to the group of
    rows with matching [label_a, label_b] columns.
    
    If a column named 'edge_area' is present, it specifies the weight that
    should be used when computing the centroid of the edge points.
    
    Any extra columns are passed on in the output.
    """
    if 'edge_area' not in all_edges_df:
        all_edges_df = all_edges_df.copy()
        all_edges_df['edge_area'] = np.int32(1)

    orig_columns = all_edges_df.columns

    all_edges_df['cz'] = all_edges_df.eval('z * edge_area')
    all_edges_df['cy'] = all_edges_df.eval('y * edge_area')
    all_edges_df['cx'] = all_edges_df.eval('x * edge_area')

    centroids_df = ( all_edges_df[['label_a', 'label_b', 'cz', 'cy', 'cx', 'edge_area']]
                       .groupby(['label_a', 'label_b']).sum() )

    centroids_df['cz'] = centroids_df.eval('cz / edge_area')
    centroids_df['cy'] = centroids_df.eval('cy / edge_area')
    centroids_df['cx'] = centroids_df.eval('cx / edge_area')

    all_edges_df.drop(['cz', 'cy', 'cx', 'edge_area'], axis=1, inplace=True)
    all_edges_df = all_edges_df.merge(centroids_df, on=['label_a', 'label_b'], how='left')
    all_edges_df['distance'] = all_edges_df.eval('sqrt( (z-cz)**2 + (y-cy)**2 + (x-cx)**2 )')

    min_distance_df = all_edges_df[['label_a', 'label_b', 'distance']].groupby(['label_a', 'label_b']).idxmin()
    min_distance_df.rename(columns={'distance': 'best_row'}, inplace=True)
    
    best_edges_df = all_edges_df.loc[min_distance_df['best_row'], orig_columns]
    return best_edges_df
