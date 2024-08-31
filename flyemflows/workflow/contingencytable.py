import os
import copy
import pickle
import logging

import numpy as np
import pandas as pd
import dask.bag as db
import pyarrow.feather as feather

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

    WARNING:
        The final step of this workflow is to compute a gigantic pandas aggregation.
        Your client (driver) machine will require lots of RAM, even though it won't be
        doing much during most of the computation before the very end.
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
            "left-subset-labels": {
                **BodyListSchema,
                "description": "If provided, limit the computation to the rows that correspond\n"
                               "to this set of labels AND the set of labels they overlap with\n"
                               "on the right-hand side.\n"
                               "Every voxel of the listed left-hand labels will be counted, \n"
                               "and for the bodies they overlap with, every voxel will be counted.\n"
                               "But objects on the left which are NOT in the listed subset will not\n"
                               "necessarily be fully accounted for in the results.\n"
                               "NOTE: This is done independently at the block-level, which means labels\n"
                               "      from the right-hand volume which overlap a subset-label in the left\n"
                               "      volume in one block are not guaranteed to be returned from different\n"
                               "      blocks, if that right-hand label doesn't overlap a left-hand label \n"
                               "      in that block.",
            },
            "min-overlap-size": {
                "description": "Discard result rows for overlapping regions smaller than this (Note: filtering is per-block).",
                "type": "integer",
                "default": 0
            },
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
                "default": 1000
            }
        }
    }

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
        sparse_fetch = not options["skip-sparse-fetch"]
        min_overlap = options["min-overlap-size"]

        # Boxes are determined by the left volume/labels/roi
        boxes = self.init_boxes( left_service,
                                 sparse_fetch and set(left_subset_labels),
                                 left_roi )

        def _contingency_table(box):
            left_vol = left_service.get_subvolume(box)
            right_vol = right_service.get_subvolume(box)

            table = contingency_table(left_vol, right_vol)
            table = table.sort_index().reset_index()

            # Compute sizes before filtering
            left_sizes = table.groupby('left')['voxel_count'].sum()
            right_sizes = table.groupby('right')['voxel_count'].sum()

            if len(left_subset_labels) > 0:
                # We keep rows if they match either of these criteria:
                #   1. they touch a left-subset label
                #   2. they touch a left label that intersects with one
                #      of the right labels from criteria 1.
                keep_left = left_sizes.index.intersection(left_subset_labels)     # noqa
                keep_right = table.query('left in @keep_left')['right'].unique()  # noqa
                table = table.query('left in @keep_left or right in @keep_right')

            if min_overlap > 1:
                table = table.query('voxel_count >= @min_overlap')

            left_sizes = left_sizes.loc[table['left'].unique()].reset_index()
            right_sizes = right_sizes.loc[table['right'].unique()].reset_index()

            return table, left_sizes, right_sizes

        batch_tables = []
        batch_left_sizes = []
        batch_right_sizes = []
        batches = iter_batches(boxes, options["batch-size"])
        logger.info(f"Computing contingency tables for {len(boxes)} bricks in total.")
        logger.info(f"Processing {len(batches)} batches of {options['batch-size']} bricks each.")

        os.makedirs('batch-tables')
        for batch_index, batch_boxes in enumerate(batches):
            with Timer(f"Batch {batch_index}: Computing tables", logger):
                # Aim for 4 partitions per worker
                total_cores = sum( self.client.ncores().values() )
                results = (db.from_sequence(batch_boxes, npartitions=4*total_cores)
                            .map(_contingency_table)
                            .compute())

                tables, left_sizes, right_sizes = zip(*results)
                table = pd.concat(tables, ignore_index=True).sort_values(['left', 'right'], ignore_index=True)
                table = table.groupby(['left', 'right'], as_index=False, sort=False)['voxel_count'].sum()

                left_sizes = pd.concat(left_sizes, ignore_index=True).groupby('left')['voxel_count'].sum().reset_index()
                right_sizes = pd.concat(right_sizes, ignore_index=True).groupby('right')['voxel_count'].sum().reset_index()

                feather.write_feather(table.reset_index(), f'batch-tables/batch-{batch_index}-table.feather')
                feather.write_feather(left_sizes, f'batch-tables/batch-{batch_index}-left-sizes.feather')
                feather.write_feather(right_sizes, f'batch-tables/batch-{batch_index}-right-sizes.feather')

                batch_tables.append(table)
                batch_left_sizes.append(left_sizes)
                batch_right_sizes.append(right_sizes)

        with Timer("Constructing final tables", logger):
            final_table = pd.concat(batch_tables, ignore_index=True).sort_values(['left', 'right']).reset_index(drop=True)
            final_table = final_table.groupby(['left', 'right'], as_index=False, sort=False)['voxel_count'].sum()

            final_left_sizes = pd.concat(batch_left_sizes, ignore_index=True).groupby('left')['voxel_count'].sum()
            final_right_sizes = pd.concat(batch_right_sizes, ignore_index=True).groupby('right')['voxel_count'].sum()

        def dump_table(table, p):
            with Timer(f"Writing {p}", logger):
                feather.write_feather(table, p)

        # feather doesn't write the index, so be sure to reset it if needed.
        dump_table(final_table, 'contingency_table.feather')
        dump_table(final_left_sizes.reset_index(), 'left_sizes.feather')
        dump_table(final_right_sizes.reset_index(), 'right_sizes.feather')

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
                if len(set(scale)) > 1:
                    raise NotImplementedError("FIXME: Can't use anisotropic scaled volume with an roi")

                scale = scale[0]
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
