import os
import copy
import logging

import h5py
import numpy as np
import pandas as pd

import dask.bag

from neuclease.util import (Timer, block_stats_for_volume, BLOCK_STATS_DTYPES, boxes_from_grid,
                            extract_subvol, overwrite_subvol, box_shape)
from neuclease.dvid import fetch_roi

from dvid_resource_manager.client import ResourceManagerClient

from ..util import replace_default_entries, COMPRESSION_METHODS, upsample
from ..volumes import ( VolumeService, DvidSegmentationVolumeSchema, DvidVolumeService )

from . import Workflow
from neuclease.util.util import iter_batches

logger = logging.getLogger(__name__)

SEGMENT_STATS_COLUMNS = ['segment', 'voxel_count', 'bounding_box_start', 'bounding_box_stop'] #, 'block_list']

class MaskSegmentation(Workflow):
    """
    Workflow to "mask out" a portion of a DVID segmentation volume.
    """

    OptionsSchema = {
        "type": "object",
        "additionalProperties": False,
        "default": {},
        "properties": {
            "block-statistics-file": {
                "description": "Where to store block statistics for the ERASED voxels.\n"
                               "This statistics in this file can't be translated to LabelIndexes directly.\n"
                               "They must be subtraced from the pre-existing label indexes.\n"
                               "If the file already exists, it will be appended to (for restarting from a failed job).\n",
                "type": "string",
                "default": "erased-block-statistics.h5"
            },
            "pyramid-depth": {
                "description": "Number of pyramid levels to update \n"
                               "(-1 means choose automatically, 0 means no pyramid,\n"
                               "in which case you'll need to generate the higher pyramid scales yourself.)\n",
                "type": "integer",
                "default": -1 # automatic by default
            },
            "download-pre-downsampled": {
                "description": "Instead of downsampling the data, just download the pyramid from the server (if it's available).\n"
                               "Will not work unless you add the 'available-scales' setting to the input service's geometry config.",
                "type": "boolean",
                "default": False
            },
            "brick-compression": {
                "description": "Internally, downloaded bricks will be stored in a compressed format.\n"
                               "This setting specifies the compression scheme to use.\n"
                               f"Options: {COMPRESSION_METHODS}"
                               "Note: This affects only in-memory storage while the workflow is running.\n"
                               "      It does NOT affect the compression used in DVID.\n",
                "type": "string",
                "enum": COMPRESSION_METHODS,
                "default": "lz4_2x"
            },
            "mask-roi": {
                "description": "Name of the ROI to use as the mask, read from the INPUT uuid.\n"
                               "Everything within this ROI will be REMOVED unless 'invert-mask' is also specified.\n",
                "type": "string"
                # no default
            },
            "invert-mask": {
                "description": "If True, mask out everything EXCEPT the given roi mask.",
                "type": "boolean",
                "default": False
            },
            "batch-size": {
                "description": "Blocks of segmentation will be processed in batches. This specifies the batch size.",
                "type": "integer",
                "default": 512
            }
        }
    }
    
    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": DvidSegmentationVolumeSchema, # Only dvid sources are supported.
        "output": DvidSegmentationVolumeSchema,
        "masksegmentation" : OptionsSchema
    })

    @classmethod
    def schema(cls):
        return MaskSegmentation.Schema


    def execute(self):
        self._sanitize_config()
        self._init_services()
        self._init_stats_file()
        
        options = self.config["masksegmentation"]
                
        mask, mask_box = self._init_mask()
        brick_boxes = boxes_from_grid(self.input_service.bounding_box_zyx, self.input_service.preferred_message_shape, clipped=True)

        with Timer("Preparing bricks", logger):
            boxes_and_masks = []
            for box in brick_boxes:
                mask_block_box = (box // (2**5)) - mask_box[0]
                mask_block = extract_subvol(mask, mask_block_box)
                if mask_block.any():
                    boxes_and_masks.append((box, mask_block))
        
        batches = iter_batches(boxes_and_masks, options["batch-size"])
        logger.info(f"Processing {len(batches)} batches")

        for batch_index, batch_boxes_and_masks in enumerate(batches):
            with Timer(f"Batch {batch_index}", logger):
                self._execute_batch(batch_index, batch_boxes_and_masks)


    def _execute_batch(self, batch_index, boxes_and_masks):
        input_service = self.input_service
        output_service = self.output_service
        
        def overwrite_box(box, lowres_mask):
            assert lowres_mask.any(), \
                "This function is supposed to be called on bricks that actually need masking"
            mask = upsample(lowres_mask, 2**5)
            seg = input_service.get_subvolume(box)

            seg_to_erase = seg.copy()
            seg_to_erase[~mask] = 0
            
            block_shape = 3*(input_service.block_width,)
            erased_stats_df = block_stats_for_volume(block_shape, seg_to_erase, box)
            del seg_to_erase
            
            new_seg = seg.copy()
            new_seg[mask] = 0
            assert not (new_seg == seg).all()

            output_service.write_subvolume(new_seg, box[0], scale=0)
            return erased_stats_df
        
        with Timer(f"Batch {batch_index:02d}: Processing blocks", logger):
            boxes_and_masks = dask.bag.from_sequence(boxes_and_masks, partition_size=1)
            erased_stats = boxes_and_masks.starmap(overwrite_box).compute()

        with Timer(f"Batch {batch_index:02d}: Combining statistics", logger):
            erased_stats_df = pd.concat(erased_stats)

        with Timer(f"Batch {batch_index:02d}: Writing statistics", logger):
            self._append_erased_statistics(erased_stats_df)

    
    def _sanitize_config(self):
        """
        Replace a few config values with reasonable defaults if necessary.
        """
        options = self.config["masksegmentation"]
        

    def _init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        
        Also check the service configurations for errors.
        """
        input_config = self.config["input"]
        output_config = self.config["output"]
        mgr_options = self.config["resource-manager"]
        options = self.config["masksegmentation"]

        self.mgr_client = ResourceManagerClient( mgr_options["server"], mgr_options["port"] )
        self.input_service = VolumeService.create_from_config( input_config, self.mgr_client )
        
        assert isinstance(self.input_service.base_service, DvidVolumeService)
        assert self.input_service.base_service.supervoxels, \
            'DVID input service config must use "supervoxels: true"'

        assert not output_config["dvid"]["create-if-necessary"], \
            "This workflow is designed to write to pre-existing DVID instances, not create them from scratch."

        # Replace 'auto' dimensions with input bounding box
        replace_default_entries(output_config["geometry"]["bounding-box"], self.input_service.bounding_box_zyx[:, ::-1])
        self.output_service = VolumeService.create_from_config( output_config, self.mgr_client )
        output_service = self.output_service
        assert isinstance(output_service.base_service, DvidVolumeService)
        assert output_service.base_service.supervoxels, \
            'DVID output service config must use "supervoxels: true"'

        assert output_service.disable_indexing, \
            "During ingestion, indexing should be disabled.\n" \
            "Please add 'disable-indexing':true to your output dvid config."

        logger.info(f"Output bounding box (xyz) is: {output_service.bounding_box_zyx[:,::-1].tolist()}")
        
        assert not (self.input_service.bounding_box_zyx % self.input_service.block_width).any(), \
            "Input bounding box must be a multiple of the block width"
        
        assert (self.input_service.bounding_box_zyx == self.output_service.bounding_box_zyx).all(), \
            "Input bounding box and output bounding box must be identical."

        # FIXME: output message shape should match input message shape
        assert not any(np.array(output_service.preferred_message_shape) % output_service.block_width), \
            "Output message-block-shape should be a multiple of the block size in all dimensions."


    def _init_mask(self):
        options = self.config["masksegmentation"]
        roi = options["mask-roi"]
        invert_mask = options["invert-mask"]

        seg_box_s5 = np.ceil(self.input_service.bounding_box_zyx / 2**5).astype(np.int32) 
        roi_mask, _ = fetch_roi(self.input_service.server, self.input_service.uuid, roi, format='mask', mask_box=seg_box_s5)
        
        if invert_mask:
            # Initialize the mask with entire segmentation at scale 5,
            # then subtract the roi from it.
            boxes = [*boxes_from_grid(seg_box_s5, (64, 64, 2048), clipped=True)]
            
            input_service = self.input_service
            def fetch_seg_mask_s5(box_s5):
                seg_s5 = input_service.get_subvolume(box_s5, scale=5)
                return box_s5, (seg_s5 != 0)
            
            boxes_and_mask = dask.bag.from_sequence(boxes, 1).map(fetch_seg_mask_s5).compute()
            
            seg_mask = np.zeros(box_shape(seg_box_s5), bool)
            for box_s5, box_mask in boxes_and_mask:
                overwrite_subvol(seg_mask, box_s5, box_mask)
            
            seg_mask[roi_mask] = False
            roi_mask = seg_mask
                
        return roi_mask, seg_box_s5


    def _init_stats_file(self):
        stats_path = self.config["masksegmentation"]["block-statistics-file"]
        if os.path.exists(stats_path):
            logger.info(f"Block statistics already exists: {stats_path}")
            logger.info(f"Will APPEND to the pre-existing statistics file.")
            return

        # Initialize a 0-entry 1D array with the correct (structured) dtype
        with h5py.File(stats_path, 'w') as f:
            f.create_dataset('stats', shape=(0,), maxshape=(None,), chunks=True, dtype=list(BLOCK_STATS_DTYPES.items()))


    def _append_erased_statistics(self, stats_df):
        """
        Append the rows of the given slab statistics DataFrame to the output statistics file.
        No attempt is made to drop duplicate rows
        (e.g. if you started from pre-existing statistics and the new
        bounding-box overlaps with the previous run's).
        
        Args:
            slab_stats_df: DataFrame to be appended to the stats file,
                           with columns and dtypes matching BLOCK_STATS_DTYPES
        """
        assert list(stats_df.columns) == list(BLOCK_STATS_DTYPES.keys())
        stats_path = self.config["masksegmentation"]["block-statistics-file"]

        with h5py.File(stats_path, 'a') as f:
            orig_len = len(f['stats'])
            new_len = orig_len + len(stats_df)
            f['stats'].resize((new_len,))
            f['stats'][orig_len:new_len] = stats_df.to_records()


