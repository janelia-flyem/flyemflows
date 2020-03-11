import copy
import logging

import numpy as np
import pandas as pd
import dask.bag as db

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, contingency_table, round_box, SparseBlockMask, boxes_from_grid, iter_batches
from neuclease.dvid import fetch_roi

from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService, ScaledVolumeService
from .util.config_helpers import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)

class ContingencyTable(Workflow):
    """
    Given two segmentation volumes ("left" and "right"),
    compute the contingency table, i.e the table of overlapping
    labels and the size (voxel count) of each unique overlapping
    left/right pair.
    """
    ContingencyTableOptionsSchema = \
    {
        "type": "object",
        "description": "Settings specific to the ContingencyTable workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "left-roi": {
                "description": "Limit analysis to bricks that intersect the given ROI, \n"
                               "which must come from the same DVID node as the LEFT input source.\n"
                               "(Only valid when the left input is a DVID source.)",
                "type": "string",
                "default": ""
            },
            "left-subset-labels": BodyListSchema,
            "skip-sparse-fetch": {
                "description": "If True, do not attempt to fetch the sparsevol-coarse for the given subset-labels.\n"
                               "Just fetch the entire bounding-box.\n",
                "type": "boolean",
                "default": False
            },
            "batch-size": {
                "description": "Brick-wise contingency tables will be computed in batches.\n"
                               "This setting specifies the number of bricks in each batch.\n",
                "type": "integer",
                "default": "1000"
            }
        }
    }

    ContingencyTableOptionsSchema["properties"]["left-subset-labels"]["description"] += (
        "If provided, only the listed labels will be analyzed.\n"
        "Other labels will be left untouched in the results.\n")


    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "left-input": SegmentationVolumeSchema,
        "right-input": SegmentationVolumeSchema,
        "contingencytable": ContingencyTableOptionsSchema
    })


    @classmethod
    def schema(cls):
        return ContingencyTable.Schema


    def execute(self):
        self.init_services()

        left_service = self.left_service
        right_service = self.right_service
        options = self.config["contingencytable"]

        left_is_supervoxels = False
        if isinstance(left_service.base_service, DvidVolumeService):
            left_is_supervoxels = left_service.base_service.supervoxels

        left_roi = options["left-roi"]
        left_subset_labels = load_body_list(options["left-subset-labels"], left_is_supervoxels)
        left_subset_labels = set(left_subset_labels)
        sparse_fetch = not options["skip-sparse-fetch"]
        
        # Boxes are determined by the left volume/labels/roi
        boxes = self.init_boxes( left_service,
                                 sparse_fetch and left_subset_labels,
                                 left_roi )

        def _contingency_table(box):
            left_vol = left_service.get_subvolume(box)
            right_vol = right_service.get_subvolume(box)
            table = contingency_table(left_vol, right_vol)
            return table.reset_index()

        batch_tables = []
        for batch_index, batch_boxes in enumerate(iter_batches(boxes, options["batch-size"])):
            with Timer(f"Batch {batch_index}: Computing tables", logger):
                tables = db.from_sequence(batch_boxes).map(_contingency_table).compute()
            
            with Timer(f"Batch {batch_index}: Combining tables", logger):
                table = pd.concat(tables, ignore_index=True).sort_values(['left', 'right']).reset_index(drop=True)
                table = table.groupby(['left', 'right'], as_index=False, sort=False)['voxel_count'].sum()

            batch_tables.append(table)

        with Timer("Constructing final table", logger):
            final_table = pd.concat(batch_tables, ignore_index=True).sort_values(['left', 'right']).reset_index(drop=True)
            final_table = final_table.groupby(['left', 'right'], as_index=False, sort=False)['voxel_count'].sum()

        with Timer("Writing contingency_table.npy", logger):
            np.save('contingency_table.npy', final_table.to_records(index=False))
        

    def init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        """
        left_config = self.config["left-input"]
        right_config = self.config["right-input"]
        mgr_config = self.config["resource-manager"]

        self.resource_mgr_client = ResourceManagerClient( mgr_config["server"], mgr_config["port"] )
        self.left_service = VolumeService.create_from_config( left_config, self.resource_mgr_client )
        self.right_service = VolumeService.create_from_config( right_config, self.resource_mgr_client )

        if (self.left_service.bounding_box_zyx != self.right_service.bounding_box_zyx).any():
            raise RuntimeError("Your left and right input volumes do not have the same bounding box.  Please specify explicit bounding boxes.")

        logger.info(f"Bounding box: {self.left_service.bounding_box_zyx[:,::-1].tolist()}")

        if (self.left_service.preferred_message_shape != self.right_service.preferred_message_shape).any():
            raise RuntimeError("Your left and right input volumes must use the same message-block-shape.")


    def init_boxes(self, volume_service, subset_labels, roi):
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
        elif subset_labels:
            try:
                sbm = volume_service.sparse_block_mask_for_labels([*subset_labels])
                if ((sbm.box[1] - sbm.box[0]) == 0).any():
                    raise RuntimeError("Could not find sparse masks for any of the subset-labels")
            except NotImplementedError:
                sbm = None

        if sbm is None:
            boxes = boxes_from_grid(volume_service.bounding_box_zyx,
                                    volume_service.preferred_message_shape,
                                    clipped=True)
            return np.array([*boxes])
        else:
            return sbm.sparse_boxes(brick_shape)
