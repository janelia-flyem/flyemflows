import os
import copy
import logging

import h5py
import vigra
import numpy as np
import pandas as pd
from skimage.util import view_as_blocks

import dask.bag

from neuclease.util import (Timer, block_stats_for_volume, BLOCK_STATS_DTYPES, boxes_from_grid, iter_batches,
                            extract_subvol, overwrite_subvol, box_shape, round_box, compute_nonzero_box)
from neuclease.dvid import fetch_instance_info, fetch_roi, encode_labelarray_blocks, post_labelmap_blocks

from dvid_resource_manager.client import ResourceManagerClient

from ..util import replace_default_entries, upsample
from ..volumes import ( VolumeService, DvidSegmentationVolumeSchema, DvidVolumeService )

from . import Workflow

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
            "min-pyramid-scale": {
                "type": "integer",
                "default": 0,
                "minValue": 0
            },
            "max-pyramid-scale": {
                "type": "integer",
                "default": -1, # choose automatically
                "maxValue": 10
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
            "dilate-mask": {
                "description": "If non-zero, dilate the ROI mask by the given radius (at scale 5)",
                "type": "integer",
                "default": 0
            },
            "erode-mask": {
                "description": "If non-zero, erode the ROI mask by the given radius (at scale 5)",
                "type": "integer",
                "default": 0
            },
            "batch-size": {
                "description": "Blocks of segmentation will be processed in batches. This specifies the batch size.",
                "type": "integer",
                "default": 512
            },
            "resume-at": {
                "description": "You can resume a failed job by specifying which scale/batch to start\n"
                               "with (assuming you haven't changed any other config settings).\n",
                "type": "object",
                "default": {"scale": 0, "batch-index": 0},
                "properties": {
                    "scale": {
                        "type": "integer",
                        "default": 0,
                        "minValue": 0
                    },
                    "batch-index": {
                        "type": "integer",
                        "default": 0,
                        "minValue": 0
                    }
                }
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
        self._init_services()
        self._sanitize_config()
        self._init_stats_file()

        options = self.config["masksegmentation"]
        min_scale = options["min-pyramid-scale"]
        max_scale = options["max-pyramid-scale"]

        starting_scale = options["resume-at"]["scale"]
        starting_batch = options["resume-at"]["batch-index"]
        
        if starting_scale != 0 or starting_batch != 0:
            logger.info(f"Resuming at scale {starting_scale} batch {starting_batch}")
        starting_scale = max(min_scale, starting_scale)
        
        mask_s5, mask_box_s5 = self._init_mask()

        for scale in range(starting_scale, 1+max_scale):
            if scale != starting_scale:
                starting_batch = 0
            
            with Timer(f"Scale {scale}: Processing", logger):
                self._execute_scale(scale, starting_batch, mask_s5, mask_box_s5)


    def _execute_scale(self, scale, starting_batch, mask_s5, mask_box_s5):
        options = self.config["masksegmentation"]
        block_width = self.output_service.block_width
        
        def scale_box(box, scale):
            # Scale down, then round up to the nearest multiple of the block width
            box = np.ceil(box / 2**scale).astype(np.int32)
            return round_box(box, block_width)

        # bounding box of the segmentation at the current scale.
        bounding_box = scale_box(self.input_service.bounding_box_zyx, scale)

        # Don't make bricks that are wider than the bounding box at this scale
        brick_shape = np.minimum(self.input_service.preferred_message_shape, bounding_box[1])
        assert not (brick_shape % block_width).any()
        
        brick_boxes = boxes_from_grid(bounding_box, brick_shape, clipped=True)

        with Timer(f"Scale {scale}: Preparing bricks", logger):
            boxes_and_masks = []
            for box in brick_boxes:
                mask_block_box = ((box // 2**(5-scale)) - mask_box_s5[0])
                mask_block_box = mask_block_box.astype(np.int32) # necessary when scale is > 5
                mask_block_s5 = np.zeros(box_shape(mask_block_box), bool)
                mask_block_s5 = extract_subvol(mask_s5, mask_block_box)
                if mask_block_s5.any():
                    boxes_and_masks.append((box, mask_block_s5))
        
        batches = [*iter_batches(boxes_and_masks, options["batch-size"])]

        if starting_batch == 0:
            logger.info(f"Scale {scale}: Processing {len(batches)} batches")
        else:
            logger.info(f"Scale {scale}: Processing {len(batches) - starting_batch} "
                        f"remaining batches from {len(batches)} original batches")

            assert starting_batch < len(batches), \
                f"Can't start at batch {starting_batch}; there are only {len(batches)} in total."
            batches = batches[starting_batch:]
            
        for batch_index, batch_boxes_and_masks in enumerate(batches, start=starting_batch):
            with Timer(f"Scale {scale}: Batch {batch_index:02d}", logger):
                self._execute_batch(scale, batch_index, batch_boxes_and_masks)


    def _execute_batch(self, scale, batch_index, boxes_and_masks):
        input_service = self.input_service
        output_service = self.output_service
        block_width = output_service.block_width
        
        def overwrite_box(box, lowres_mask):
            assert not (box[0] % block_width).any()
            assert lowres_mask.any(), \
                "This function is supposed to be called on bricks that actually need masking"

            # Crop box and mask to only include the extent of the masked voxels
            nonzero_mask_box = compute_nonzero_box(lowres_mask)
            nonzero_mask_box = round_box(nonzero_mask_box, (block_width * 2**scale) // 2**5)
            lowres_mask = extract_subvol(lowres_mask, nonzero_mask_box)
            
            box = box[0] + (nonzero_mask_box * 2**(5-scale))
            box = box.astype(np.int32)

            if scale <= 5:
                mask = upsample(lowres_mask, 2**(5-scale))
            else:
                # Downsample, but favor UNmasked voxels
                mask = ~view_as_blocks(~lowres_mask, 3*(2**(scale-5),)).any(axis=(3,4,5))
            
            old_seg = input_service.get_subvolume(box, scale)

            new_seg = old_seg.copy()
            new_seg[mask] = 0
            
            if (new_seg == old_seg).all():
                # It's possible that there are no changed voxels, but only
                # at high scales where the masked voxels were downsampled away.
                #
                # So if the original downscale pyramids are perfect,
                # then the following assumption ought to hold.
                #
                # But I'm commenting it out in case the DVID pyramid at scale 5
                # isn't pixel-perfect in some places.
                #
                # assert scale > 5
                
                return None
            
            def post_changed_blocks(old_seg, new_seg):
                # If we post the whole volume, we'll be overwriting blocks that haven't changed,
                # wasting space in DVID (for duplicate blocks stored in the child uuid).
                # Instead, we need to only post the blocks that have changed.
    
                # So, can't just do this:
                # output_service.write_subvolume(new_seg, box[0], scale)
    
                seg_diff = (old_seg != new_seg)
                block_diff = view_as_blocks(seg_diff, 3*(block_width,))
    
                changed_block_map = block_diff.any(axis=(3,4,5)).nonzero()
                changed_block_corners = box[0] + np.transpose(changed_block_map) * block_width
    
                changed_blocks = view_as_blocks(new_seg, 3*(block_width,))[changed_block_map]
                encoded_blocks = encode_labelarray_blocks(changed_block_corners, changed_blocks)
                
                mgr = output_service.resource_manager_client
                with mgr.access_context(output_service.server, True, 1, changed_blocks.nbytes):
                    post_labelmap_blocks( *output_service.instance_triple, None, encoded_blocks, scale,
                                          downres=False, noindexing=True, throttle=False,
                                          is_raw=True )

            assert not (box % block_width).any(), \
                "Should not write partial blocks"

            post_changed_blocks(old_seg, new_seg)
            del new_seg

            if scale != 0:
                # Don't collect statistics for higher scales
                return None
            
            erased_seg = old_seg.copy()
            erased_seg[~mask] = 0
            
            block_shape = 3*(input_service.block_width,)
            erased_stats_df = block_stats_for_volume(block_shape, erased_seg, box)
            return erased_stats_df
        
        with Timer(f"Scale {scale}: Batch {batch_index:02d}: Processing blocks", logger):
            boxes_and_masks = dask.bag.from_sequence(boxes_and_masks, partition_size=1)
            erased_stats = boxes_and_masks.starmap(overwrite_box).compute()

        if scale == 0:
            with Timer(f"Scale {scale}: Batch {batch_index:02d}: Combining statistics", logger):
                erased_stats_df = pd.concat(erased_stats)
    
            with Timer(f"Scale {scale}: Batch {batch_index:02d}: Writing statistics", logger):
                self._append_erased_statistics(erased_stats_df)

    
    def _sanitize_config(self):
        """
        Replace a few config values with reasonable defaults if necessary.
        Must be called after the input/output services are initialized.
        """
        options = self.config["masksegmentation"]
        
        if options["max-pyramid-scale"] == -1:
            info = fetch_instance_info(*self.output_service.instance_triple)
            existing_depth = int(info["Extended"]["MaxDownresLevel"])
            options["max-pyramid-scale"] = existing_depth

        # FIXME
        #if options["resume-at"]["scale"] < options["min-pyramid-scale"]:
        #    raise RuntimeError("Your 'resume-at' scale seems not to agree with your "
        #                       "original min-pyramid-scale. Is this really a resumed job?")

        if options["dilate-mask"] > 0 and options["erode-mask"] > 0:
            raise RuntimeError("Can't dilate mask and erode it, too.  Choose one or the other.")

    def _init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        
        Also check the service configurations for errors.
        """
        input_config = self.config["input"]
        output_config = self.config["output"]
        mgr_options = self.config["resource-manager"]

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
        max_scale = options["max-pyramid-scale"]
        dilation_radius = options["dilate-mask"]
        erosion_radius = options["erode-mask"]
        
        block_width = self.output_service.block_width

        # Select a mask_box that's large enough to divide evenly into the
        # block width even when reduced to the highest scale we'll be processing.
        seg_box = round_box(self.input_service.bounding_box_zyx, block_width * 2**max_scale)
        seg_box_s5 = round_box(seg_box, 2**5) // (2**5)

        with Timer(f"Loading ROI '{roi}'", logger):
            roi_mask, _ = fetch_roi(self.input_service.server, self.input_service.uuid, roi, format='mask', mask_box=seg_box_s5)
        
        if invert_mask:
            with Timer("Inverting mask", logger):
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

        assert not (dilation_radius and erosion_radius)
        if dilation_radius > 0:
            with Timer(f"Dilating mask by {dilation_radius}", logger):
                roi_mask = vigra.filters.multiBinaryDilation(roi_mask, dilation_radius)
        if erosion_radius > 0:
            with Timer(f"Eroding mask by {erosion_radius}", logger):
                roi_mask = vigra.filters.multiBinaryErosion(roi_mask, erosion_radius)

        # Downsample the roi_mask to dvid-block resolution, just to see how many blocks it touches. 
        block_mask = view_as_blocks(roi_mask, (2,2,2)).any(axis=(3,4,5))
        blocks_touched = block_mask.sum()
        voxel_total = blocks_touched * (block_width**3)
        logger.info(f"Mask touches {blocks_touched} blocks ({voxel_total / 1e9:.1f} Gigavoxels)")

        return roi_mask, seg_box_s5


    def _init_stats_file(self):
        options = self.config["masksegmentation"]

        if options["resume-at"]["scale"] > 0 or options["min-pyramid-scale"] > 0:
            logger.info("Not processing scale 0, so not computing batch statistics.")
            return
        
        stats_path = options["block-statistics-file"]
        if os.path.exists(stats_path):
            logger.info(f"Block statistics already exists: {stats_path}")
            if options["resume-at"]["batch-index"] == 0:
                raise RuntimeError("Refusing to append to a pre-existing block statistics file")
            logger.info(f"Resuming from a previous workload.  Will APPEND to the pre-existing statistics file.")
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
