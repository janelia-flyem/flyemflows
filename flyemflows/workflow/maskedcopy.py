import copy
import logging

import numpy as np
import pandas as pd
import dask.bag as db

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, round_box, SparseBlockMask, boxes_from_grid, iter_batches
from neuclease.dvid import fetch_roi

from ..util import replace_default_entries
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService, ScaledVolumeService
from . import Workflow

logger = logging.getLogger(__name__)


class MaskedCopy(Workflow):
    """
    Given a segmentation volume and a mask volume,
    copy the segmentation, but only under the voxels that are nonzero in the mask.
    All other voxels will be written as zeros, or omitted entirely (since they are
    presumed to be zeros in the output by default.)

    Note:
        This masks out the segmentation based on a high-res volume of the same shape.
        To mask out a segmentation based on a DVID ROI, see the MaskSegmentation workflow.
        (The "roi" option below only applies to the portion of the volume that will be analyzed,
        but is not otherwise used to define the mask.)
    """
    MaskedCopyOptionsSchema = \
    {
        "type": "object",
        "description": "Settings specific to the MaskedCopy workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "roi": {
                "description": "Optional. Limit analysis to bricks that intersect the given ROI, \n"
                               "which must come from the same DVID node as the segmentation source, not the mask source.\n"
                               "Note: This is solely for avoiding unnecessary computation.  Does not affect the \n"
                               "(Only valid when the input is a DVID source.)",
                "type": "string",
                "default": ""
            },
            "batch-size": {
                "description": "The result is computed brickwise, in batches of bricks.\n"
                               "This setting specifies the number of bricks in each batch.\n",
                "type": "integer",
                "default": 1000
            },
            "restart-at-batch": {
                "description": "If you're restarting the job for some reason and\n"
                               "you want to skip the first N batches, set this.\n",
                "type": "integer",
                "default": 0
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,
        "mask": SegmentationVolumeSchema,
        "output": SegmentationVolumeSchema,
        "maskedcopy": MaskedCopyOptionsSchema
    })

    @classmethod
    def schema(cls):
        return MaskedCopy.Schema

    def execute(self):
        input_service, mask_service, output_service = self.init_services()
        options = self.config["maskedcopy"]

        # Boxes are determined by the left volume/labels/roi
        boxes = self.init_boxes( input_service, options["roi"] )

        def _masked_copy(box):
            seg_vol = input_service.get_subvolume(box)
            mask_vol = mask_service.get_subvolume(box).astype(bool)
            seg_vol[~mask_vol] = 0
            output_service.write_subvolume(seg_vol, box[0])
            return (*box[0], mask_vol.sum())

        batches = iter_batches(boxes, options["batch-size"])
        logger.info(f"Performing masked copy of {len(boxes)} bricks in total.")
        logger.info(f"Processing {len(batches)} batches of {options['batch-size']} bricks each.")

        for batch_index, batch_boxes in enumerate(batches):
            if batch_index < options["restart-at-batch"]:
                logger.info("Batch {batch_index}: Skipping")
                continue

            with Timer(f"Batch {batch_index}: Copying", logger):
                # Aim for 4 partitions per worker
                total_cores = sum( self.client.ncores().values() )
                brick_counts = (db.from_sequence(batch_boxes, npartitions=4*total_cores)
                                  .map(_masked_copy)
                                  .compute())

                brick_counts_df = pd.DataFrame(brick_counts, columns=[*'zyx', 'mask_voxels'])
                brick_counts_df.to_csv(f'batch-{batch_index:03d}-brick-mask-voxels.csv', header=True, index=False)

    def init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        """
        mgr_config = self.config["resource-manager"]
        input_config = self.config["input"]
        mask_config = self.config["mask"]
        output_config = self.config["output"]

        resource_mgr_client = ResourceManagerClient( mgr_config["server"], mgr_config["port"] )
        input_service = VolumeService.create_from_config( input_config, resource_mgr_client )
        logger.info(f"Bounding box: {input_service.bounding_box_zyx[:,::-1].tolist()}")

        # Replace default entries in the output and mask bounding boxes
        replace_default_entries(mask_config["geometry"]["bounding-box"], input_service.bounding_box_zyx[:, ::-1])
        replace_default_entries(output_config["geometry"]["bounding-box"], input_service.bounding_box_zyx[:, ::-1])

        mask_service = VolumeService.create_from_config( mask_config, resource_mgr_client )
        output_service = VolumeService.create_from_config( output_config, resource_mgr_client )

        if (input_service.preferred_message_shape != mask_service.preferred_message_shape).any():
            raise RuntimeError("Your input volume and mask volume must use the same message-block-shape.")

        if isinstance(output_service.base_service, DvidVolumeService) and not output_service.base_service.write_empty_blocks:
            logger.warning("Your output config does not set write-empty-blocks: True "
                           "-- consider changing that to avoid writing needless zeros to DVID!.")

        return input_service, mask_service, output_service

    def init_boxes(self, volume_service, roi):
        sbm = None
        if roi:
            base_service = volume_service.base_service
            assert isinstance(base_service, DvidVolumeService), \
                "Can't specify an ROI unless you're using a dvid input"

            assert isinstance(volume_service, (ScaledVolumeService, DvidVolumeService)), \
                "The 'roi' option doesn't support adapters other than 'rescale-level'"
            scale = 0
            if isinstance(volume_service, ScaledVolumeService):
                scale = volume_service.scale_delta
                assert scale <= 5, \
                    "The 'roi' option doesn't support volumes downscaled beyond level 5"

            server, uuid, _seg_instance = base_service.instance_triple

            brick_shape = volume_service.preferred_message_shape
            assert not (brick_shape % 2**(5-scale)).any(), \
                "If using an ROI, select a brick shape that is divisible by 32"

            seg_box = volume_service.bounding_box_zyx
            seg_box = round_box(seg_box, brick_shape)
            seg_box_s0 = seg_box * 2**scale
            seg_box_s5 = seg_box // 2**(5-scale)

            with Timer(f"Fetching mask for ROI '{roi}' ({seg_box_s0[:, ::-1].tolist()})", logger):
                roi_mask_s5, _ = fetch_roi(server, uuid, roi, format='mask', mask_box=seg_box_s5)

            # SBM 'full-res' corresponds to the input service voxels, not necessarily scale-0.
            sbm = SparseBlockMask.create_from_highres_mask(roi_mask_s5, 2**(5-scale), seg_box, brick_shape)

        if sbm is None:
            boxes = boxes_from_grid(volume_service.bounding_box_zyx,
                                    volume_service.preferred_message_shape,
                                    clipped=True)
            return np.array([*boxes])
        else:
            return sbm.sparse_boxes(brick_shape)
