import os
import copy
import logging

import h5py
import numpy as np
import pandas as pd

from neuclease.util import Timer, Grid, block_stats_for_volume, BLOCK_STATS_DTYPES

from dvid_resource_manager.client import ResourceManagerClient

from ..brick import BrickWall
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService

from . import Workflow
from .util.config_helpers import BodyListSchema, load_body_list

logger = logging.getLogger(__name__)

class SparseBlockstats(Workflow):
    """
    Compute block statistics which can be used later to create label indexes.
    Only a sparse subset of blocks will be processed, which is specified by providing a mask-input source,
    along with the subset of labels in that input to determine which blocks should be used.
    
    So, the 'mask-input' and 'mask-labels' determines WHERE to look, but the 'input'
    is the segmentation that is actually read from to generate the block stats.
    """
    OptionsSchema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "block-statistics-file": {
                "description": "Where to store block statistics for the INPUT segmentation\n"
                               "(but translated to output coordinates).\n"
                               "If the file already exists, it will be appended to (for restarting from a failed job).\n"
                               "Supported formats: .csv and .h5",
                "type": "string",
                "default": "block-statistics.h5"
            },
            "mask-labels": {
                "description": "Only blocks which contain the given labels from the mask-input will be processed.\n",
                **BodyListSchema
            }
        }
    }


    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,
        "mask-input": SegmentationVolumeSchema,
        "sparseblockstats" : OptionsSchema
    })


    @classmethod
    def schema(cls):
        return SparseBlockstats.Schema


    def execute(self):
        input_wall = self.init_brickwall()

        block_shape = 3*[self.config["input"]["geometry"]["block-width"]]
        def compute_stats(brick):
            return block_stats_for_volume(block_shape, brick.volume, brick.physical_box)

        with Timer("Computing block stats", logger):            
            all_stats = input_wall.bricks.map(compute_stats).compute()

        with Timer("Concatenating block stats", logger):            
            stats_df = pd.concat(all_stats, ignore_index=True)
        
        with Timer("Writing block stats", logger):            
            self.write_block_stats(stats_df)
        

    def init_brickwall(self):
        input_config = self.config["input"]
        mask_input_config = self.config["mask-input"]
        mgr_config = self.config["resource-manager"]
        options = self.config["sparseblockstats"]
        
        resource_mgr_client = ResourceManagerClient( mgr_config["server"], mgr_config["port"] )
        input_service = VolumeService.create_from_config( input_config, resource_mgr_client )
        mask_service = VolumeService.create_from_config( mask_input_config, resource_mgr_client )
        
        assert (input_service.preferred_message_shape == mask_service.preferred_message_shape).all(), \
            "This workflow assumes that the input and the mask-input use the same brick grid."

        assert not (input_service.preferred_message_shape % input_service.block_width).any(), \
            "input brick grid spacing must be a multipe of the input's block-width"
        assert not (mask_service.preferred_message_shape % mask_service.block_width).any(), \
            "mask brick grid spacing must be a multipe of the input's block-width"

        is_supervoxels = False
        if isinstance(mask_service.base_service, DvidVolumeService):
            is_supervoxels = mask_service.base_service.supervoxels

        # Load body list and eliminate duplicates
        subset_labels = load_body_list(options["mask-labels"], is_supervoxels)
        subset_labels = set(subset_labels)

        if not subset_labels:
            raise RuntimeError("You didn't specify any mask subset labels. "
                               "If you want to compute block stats for an entire segmentation volume, use the CopySegmentation workflow.")

        sbm = mask_service.sparse_block_mask_for_labels(subset_labels)
        if ((sbm.box[1] - sbm.box[0]) == 0).any():
            raise RuntimeError("Could not find sparse masks for any of the mask-labels")

        with Timer("Initializing BrickWall", logger):
            # Aim for 2 GB RDD partitions when loading segmentation
            GB = 2**30
            target_partition_size_voxels = 2 * GB // np.uint64().nbytes
            brickwall = BrickWall.from_volume_service(input_service, 0, None, self.client, target_partition_size_voxels, 0, sbm)

            # Pad if necessary to ensure that all fetched bricks are block-aligned
            block_shape = 3*(input_service.block_width,)
            brickwall = brickwall.fill_missing(input_service.get_subvolume, Grid(block_shape))

        return brickwall


    def init_block_stats_file(self):
        stats_path = self.config["sparseblockstats"]["block-statistics-file"]
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
        stats_path = self.config["sparseblockstats"]["block-statistics-file"]

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


