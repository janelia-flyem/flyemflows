import os
import copy
import logging

import numpy as np
import pandas as pd

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import read_csv_header, lexsort_columns, Timer, box_intersection, groupby_presorted, groupby_spans_presorted, SparseBlockMask

from ..brick import BrickWall
from ..volumes import VolumeService, SegmentationVolumeSchema
from . import Workflow

logger = logging.getLogger(__name__)

# Common CSV column types (unknown columns will have their types guessed by pandas)
CSV_TYPES = { 'x': np.int32,
              'y': np.int32,
              'z': np.int32,
              'kind': 'category',
              'conf': np.float32,
              'user': 'category',
              'label': np.uint64,
              'body': np.uint64,
              'sv': np.uint64 }


class SamplePoints(Workflow):
    """
    Workflow to read a CSV of point coordinates and sample those points from a segmentation instance.

    The volume is divided into Bricks, and the points are grouped by target brick.
    No data is fetched for Bricks that don't have points within them.
    After sampling, the results are aggregated and exported to CSV.
    
    All columns from the original CSV file are preserved, but the rows will not necessarily
    be in the same order as the input file.  They will be sorted by coordinate.
    """
    SamplePointsOptionsSchema = \
    {
        "type": "object",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "input-table": {
                "description": "Table to read points from. Must be .csv (with header!)",
                "type": "string"
            },
            "output-table": {
                "description": "Results file.  Must be .csv for now, and must contain at least columns x,y,z",
                "type": "string",
                "default": "point-samples.csv"
            },
            "rescale-points-to-level": {
                "description": "Specifies a scale (power of 2) by which to divide the loaded point coordinates before beginning the analysis.\n"
                               "Typically used if you are applying a 'rescale-level' to your input source.\n"
                               "Note: The points will appear rescaled in the output file.  The original points are not preserved.\n",
                "type": "integer",
                "default": 0
            },
            "output-column": {
                "description": "The name of the output column in the final CSV results",
                "type": "string",
                "default": "label"
            }
            # TODO:
            # - Support .npy input
            # - Support alternative column names instead of x,y,z (e.g. 'xa', 'ya', 'yb')
        }
    }

    Schema = copy.deepcopy(Workflow.BaseSchema)
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,
        "samplepoints": SamplePointsOptionsSchema
    })


    @classmethod
    def schema(cls):
        return SamplePoints.Schema


    def _sanitize_config(self):
        """
        - Normalize/overwrite certain config values
        - Check for config mistakes
        - Simple sanity checks
        """
        # Convert input/output CSV to absolute paths
        options = self.config["samplepoints"]
        header = read_csv_header(options["input-table"])
        if header is None:
            raise RuntimeError(f"Input table does not have a header row: {options['input-table']}")
        
        if set('zyx') - set(header):
            raise RuntimeError(f"Input table does not have the expected column names: {options['input-table']}")


    def execute(self):
        self._sanitize_config()

        input_config = self.config["input"]
        options = self.config["samplepoints"]
        resource_config = self.config["resource-manager"]

        resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])
        volume_service = VolumeService.create_from_config(input_config, os.getcwd(), resource_mgr_client)

        input_csv = options["input-table"]
        with Timer(f"Reading {input_csv}", logger):
            coordinate_table_df = pd.read_csv(input_csv, header=0, dtype=CSV_TYPES)
            points = coordinate_table_df[['z', 'y', 'x']].values

        rescale = options["rescale-points-to-level"]
        if rescale != 0:
            points //= 2**rescale

        # All points must lie within the input volume        
        points_box = [points.min(axis=0), 1+points.max(axis=0)]
        if (box_intersection(points_box, volume_service.bounding_box_zyx) != points_box).all():
            raise RuntimeError("The point list includes points outside of the volume bounding box.")

        with Timer("Sorting points by Brick ID", logger):
            # 'Brick ID' is defined as the divided corner coordinate 
            brick_shape = volume_service.preferred_message_shape
            brick_ids_and_points = np.concatenate( (points // brick_shape, points), axis=1 )
            lexsort_columns(brick_ids_and_points)

            brick_ids = brick_ids_and_points[: ,:3]
            points = brick_ids_and_points[:, 3:]
            
            # Extract the first row of each group to get the set of unique brick IDs
            point_group_spans = groupby_spans_presorted(brick_ids)
            point_group_starts = (start for start, stop in point_group_spans)
            unique_brick_ids = brick_ids[np.fromiter(point_group_starts, np.int32)]

        with Timer("Constructing sparse mask", logger):
            # BrickWall.from_volume_service() supports the ability to initialize a sparse RDD,
            # with only a subset of Bricks (rather than a dense RDD containing every brick
            # within the volume bounding box).
            # It requires a SparseBlockMask object indicating exactly which Bricks need to be fetched.
            brick_mask_box = np.array([unique_brick_ids.min(axis=0), 1+unique_brick_ids.max(axis=0)])

            brick_mask_shape = (brick_mask_box[1] - brick_mask_box[0])
            brick_mask = np.zeros(brick_mask_shape, bool)
            brick_mask_coords = unique_brick_ids - brick_mask_box[0]
            brick_mask[tuple(brick_mask_coords.transpose())] = True
            sbm = SparseBlockMask(brick_mask, brick_mask_box*brick_shape, brick_shape)

        with Timer("Initializing BrickWall", logger):
            # Aim for 2 GB RDD partitions when loading segmentation
            GB = 2**30
            target_partition_size_voxels = 2 * GB // np.uint64().nbytes
            brickwall = BrickWall.from_volume_service(volume_service, 0, None, self.client, target_partition_size_voxels, sbm, lazy=True)
        
        with Timer("Grouping points", logger):
            # This is faster than pandas.DataFrame.groupby() for large data
            point_groups = groupby_presorted(points, brick_ids)
            id_and_ptgroups = list(zip(unique_brick_ids, point_groups))

        with Timer("Joining point groups with bricks", logger):
            ptgroup_and_brick = brickwall.bricks.join(id_and_ptgroups,
                                                      lambda brick: tuple(brick.logical_box[0] // brick_shape),
                                                      lambda id_and_ptgroup: tuple(id_and_ptgroup[0]))
            
        with Timer("Persisting joined point groups", logger):
            # Persist and force computation before proceeding.
            ptgroup_and_brick = ptgroup_and_brick.persist()
            ptgroup_and_brick.count().compute()

        def sample_points(points_and_brick):
            """
            Given a Brick and array of points (N,3) that lie within it,
            sample labels from the points within the brick and return
            a record array containing the points and the sampled labels.
            """
            (_brick_id, points), brick = points_and_brick

            result_dtype = [('z', np.int32), ('y', np.int32), ('x', np.int32), ('label', np.uint64)]
            result = np.zeros((len(points),), result_dtype)
            result['z'] = points[:,0]
            result['y'] = points[:,1]
            result['x'] = points[:,2]

            # Make relative to brick offset
            points -= brick.physical_box[0]
            
            result['label'] = brick.volume[tuple(points.transpose())]
            return result

        with Timer("Sampling bricks", logger):
            brick_samples = ptgroup_and_brick.map(sample_points).compute()

        with Timer("Concatenating samples", logger):
            sample_table = np.concatenate(brick_samples)

        with Timer("Sorting samples", logger):
            sample_table.sort()

        output_col = options["output-column"]
        with Timer("Sorting table", logger):
            coordinate_table_df.sort_values(['z', 'y', 'x'], inplace=True)

        # Now that samples and input rows are sorted identically,
        # append the results
        coordinate_table_df[output_col] = sample_table['label']

        with Timer("Exporting samples", logger):
            coordinate_table_df.to_csv(options["output-table"], header=True, index=False)

        logger.info("DONE.")
