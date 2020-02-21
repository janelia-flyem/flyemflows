import copy
import logging

import numpy as np
import pandas as pd

import dask.bag

from neuclease.dvid import fetch_combined_roi_volume
from neuclease.util import Timer, iter_batches, extract_subvol, box_shape, box_intersection, apply_mask_for_labels
from dvid_resource_manager.client import ResourceManagerClient

from ..util import upsample
from ..volumes import VolumeService, ScaledVolumeService, DvidSegmentationVolumeSchema, DvidVolumeService
from .util.config_helpers import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)


class RoiStats(Workflow):
    """
    Simple workflow to calculate the per-ROI voxel counts for a set of bodies. 
    """
    OptionsSchema = {
        "type": "object",
        "additionalProperties": False,
        "default": {},
        "properties": {
            "rois": {
                "description": "List of ROI names. Required.",
                "type": "array",
                "items": {
                    "type": "string"
                }
                # no default
            },
            "subset-bodies": BodyListSchema,
            "batch-size": {
                "description": "Bricks of segmentation will be processed in batches. This specifies the batch size.",
                "type": "integer",
                "default": 512
            }
        }
    }
    
    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": DvidSegmentationVolumeSchema, # Only dvid sources are supported.
        "roistats" : OptionsSchema
    })


    @classmethod
    def schema(cls):
        return RoiStats.Schema


    def execute(self):
        scale = self._init_service()

        options = self.config["roistats"]
        server = self.input_service.base_service.server
        uuid = self.input_service.base_service.uuid
        rois = options["rois"]

        bodies = load_body_list(options["subset-bodies"], self.input_service.base_service.supervoxels)
        assert len(bodies) > 0, "Please provide a list of subset-bodies to process"

        bounding_box = self.input_service.bounding_box_zyx
        assert not (bounding_box % 2**(5-scale)).any(), \
            "Make sure your configured bounding box is divisible by 32px at scale 0"
        brick_shape = self.input_service.preferred_message_shape
        assert not (brick_shape % 2**(5-scale)).any(), \
            "Make sure your preferred message shape divides into 32px blocks at scale 0"

        with Timer("Fetching ROI volume", logger):
            roi_vol_s5, roi_box_s5, overlaps = fetch_combined_roi_volume(server, uuid, rois, False, bounding_box // 2**(5-scale))

        if len(overlaps) > 0:
            logger.warn(f"Some of your ROIs overlap!  Here's an incomplete list:\n{overlaps}")

        with Timer("Determining brick set", logger):
            brick_coords_df = self.input_service.sparse_brick_coords_for_labels(bodies)
            np.save('brick-coords.npy', brick_coords_df.to_records(index=False))

        with Timer(f"Preparing bricks", logger):
            boxes_and_roi_bricks = []
            for coord, labels in brick_coords_df.groupby([*'zyx'])['label'].agg(tuple).iteritems():
                box = np.array((coord, coord))
                box[1] += brick_shape
                box = box_intersection(box, bounding_box)
    
                roi_brick_box = ((box // 2**(5-scale)) - roi_box_s5[0])
                roi_brick_s5 = extract_subvol(roi_vol_s5, roi_brick_box)
                boxes_and_roi_bricks.append((box, roi_brick_s5, labels))
        
        logger.info(f"Prepared {len(boxes_and_roi_bricks)} bricks of shape {(*brick_shape[::-1],)}")
        
        all_stats = []
        batches = [*iter_batches(boxes_and_roi_bricks, options["batch-size"])]
        logger.info(f"Processing {len(batches)} batches")
        for i, batch_boxes_and_bricks in enumerate(batches):
            with Timer(f"Batch {i:02d}", logger):
                batch_stats = self._execute_batch(scale, batch_boxes_and_bricks)
                all_stats.append( batch_stats )

        all_stats = pd.concat(all_stats, ignore_index=True)
        all_stats = all_stats.groupby(['body', 'roi_id'], as_index=False)['voxels'].sum()
        
        roi_names = pd.Series(["<none>", *rois], name='roi')
        roi_names.index.name = 'roi_id'
        all_stats = all_stats.merge(roi_names, 'left', on='roi_id')
        all_stats = all_stats.sort_values(['body', 'roi_id'])
        
        if scale > 0:
            all_stats.rename(columns={'voxels': f'voxels_s{scale}'}, inplace=True)
        
        with Timer(f"Writing stats ({len(all_stats)} rows)", logger):
            np.save('roi-stats.npy', all_stats.to_records(index=False))
            all_stats.to_csv('roi-stats.csv', index=False, header=True)
        

    def _execute_batch(self, scale, batch_boxes_and_bricks):
        input_service = self.input_service
        def process_brick(box, roi_brick_s5, labels):
            roi_brick = upsample(roi_brick_s5, 2**(5-scale))
            assert (roi_brick.shape == box_shape(box)).all()

            # Download seg, but erase everything except our bodies of interest.
            # Note: Service is already configured at the right scale.
            seg_brick = input_service.get_subvolume(box)
            seg_brick = np.asarray(seg_brick, order='C')
            apply_mask_for_labels(seg_brick, labels, inplace=True)

            df = pd.DataFrame({ 'body': seg_brick.reshape(-1),
                                'roi_id': roi_brick.reshape(-1) })

            stats = (df.groupby(['body', 'roi_id'])
                       .size()
                       .rename('voxels')
                       .reset_index()
                       .query('body != 0'))

            return stats

        stats_bag = dask.bag.from_sequence(batch_boxes_and_bricks, partition_size=1 ).starmap(process_brick)
        stats = stats_bag.compute()
        stats = pd.concat(stats, ignore_index=True)
        stats = stats.groupby(['body', 'roi_id'], as_index=False)['voxels'].sum()
        return stats


    def _init_service(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        
        Also check the service configurations for errors.
        """
        input_config = self.config["input"]
        mgr_options = self.config["resource-manager"]

        self.mgr_client = ResourceManagerClient( mgr_options["server"], mgr_options["port"] )
        self.input_service = VolumeService.create_from_config( input_config, self.mgr_client )
        
        assert isinstance(self.input_service.base_service, DvidVolumeService), \
            "Only DVID sources are permitted by this workflow."

        assert not (self.input_service.bounding_box_zyx % self.input_service.block_width).any(), \
            "Input bounding box must be a multiple of the block width"

        if isinstance(self.input_service, ScaledVolumeService):
            scale = self.input_service.scale_delta
            assert scale <= 5, "Can't use rescale-level > 5 in this workflow."
            return scale
        return 0
