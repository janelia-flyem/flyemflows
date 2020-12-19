import copy
import logging

import numpy as np
import pandas as pd

import dask.bag

from neuclease.dvid import fetch_combined_roi_volume, resolve_ref
from neuclease.util import Timer, iter_batches, extract_subvol, box_shape, box_intersection, apply_mask_for_labels, boxes_from_mask
from dvid_resource_manager.client import ResourceManagerClient

from ..util import upsample
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService
from .util.config_helpers import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)


class RoiStats(Workflow):
    """
    Simple workflow to calculate the per-ROI voxel counts for a set of bodies.

    See Also:
        flyemflows/workflow/util/roistats_table.py
    """
    OptionsSchema = {
        "type": "object",
        "additionalProperties": False,
        "default": {},
        "properties": {
            "roi-server": {
                "description": "Address of the DVID server from which to read ROIs.\n"
                               "Can be omitted if the input source is a DVID instance.\n",
                "type": "string",
                "default": ""
            },
            "roi-uuid": {
                "description": "UUID from which to read ROIs.\n"
                               "Can be omitted if the input source is a DVID instance.\n",
                "type": "string",
                "default": ""
            },
            "rois": {
                "description": "List of ROI names. Required.",
                "type": "array",
                "items": {
                    "type": "string"
                }
                # no default
            },
            "analysis-scale": {
                "description": "Scale at which to perform the analysis, relative to the input data.\n"
                               "Make sure the difference between your input data scale and the ROI scale in DVID is (2**5).\n"
                               "Then use this parameter to specify a downsampling scale to use when reading the input.\n",
                "type": "integer",
                "minValue": 0,
                "maxValue": 10,
                "default": 0
            },
            "subset-bodies": {
                **BodyListSchema,
                "description": "Specify a list of labels to analyze.\n"
                               "If the input is a DVID source, the downloaded segmentation\n"
                               "will be limited to the bricks that contain these labels.\n"
                               "Otherwise, it will be limited to bricks that intersect the listed ROIs.\n",
            },
            "batch-size": {
                "description": "Bricks of segmentation will be processed in batches. This specifies the batch size.",
                "type": "integer",
                "default": 512
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,
        "roistats": OptionsSchema
    })

    @classmethod
    def schema(cls):
        return RoiStats.Schema

    def execute(self):
        self._init_service()
        options = self.config["roistats"]

        if not options["roi-server"]:
            assert isinstance(self.input_service, DvidVolumeService)
            options["roi-server"] = self.input_service.base_service.server

        if not options["roi-uuid"]:
            assert isinstance(self.input_service, DvidVolumeService)
            options["roi-uuid"] = self.input_service.base_service.uuid

        options["roi-uuid"] = resolve_ref(options["roi-server"], options["roi-uuid"])

        is_supervoxels = (isinstance(self.input_service, DvidVolumeService)
                          and self.input_service.base_service.supervoxels) # noqa
        bodies = load_body_list(options["subset-bodies"], is_supervoxels)
        assert len(bodies) > 0, "Please provide a list of subset-bodies to process"

        scale = options["analysis-scale"]
        bounding_box = self.input_service.bounding_box_zyx
        assert not (bounding_box % 2**5).any(), \
            "Make sure your configured bounding box is divisible by 32px at scale 0."
        brick_shape = self.input_service.preferred_message_shape
        assert not (brick_shape % 2**5).any(), \
            "Make sure your preferred message shape divides into 32px blocks at scale 0"

        with Timer("Fetching ROI volume", logger):
            roi_vol_s5, roi_box_s5, overlaps = fetch_combined_roi_volume( options["roi-server"],
                                                                          options["roi-uuid"],
                                                                          options["rois"],
                                                                          False,
                                                                          bounding_box // 2**5 )

        if len(overlaps) > 0:
            logger.warn(f"Some of your ROIs overlap!  Here's an incomplete list:\n{overlaps}")

        with Timer("Determining brick set", logger):
            # Determine which bricks intersect our ROIs
            roi_brick_shape = self.input_service.preferred_message_shape // 2**5
            roi_brick_boxes = boxes_from_mask((roi_vol_s5 != 0), roi_box_s5[0], roi_brick_shape, clipped=False)
            roi_brick_boxes *= 2**5
            roi_brick_boxes = box_intersection(roi_brick_boxes, self.input_service.bounding_box_zyx)

            # Non-intersecting boxes have negative shape -- drop them.
            roi_brick_boxes = roi_brick_boxes[((roi_brick_boxes[:, 1, :] - roi_brick_boxes[:, 0, :]) > 0).all(axis=1)]
            roi_brick_coords_df = pd.DataFrame(roi_brick_boxes[:, 0, :], columns=[*'zyx'])
            try:
                body_brick_coords_df = self.input_service.sparse_brick_coords_for_labels(bodies)
            except NotImplementedError:
                # Use all bricks in the ROIs, and use the special label -1 to
                # indicate that all bodies in the list might be found there.
                # (See below.)
                brick_coords_df = roi_brick_coords_df
                brick_coords_df['label'] = -1
            else:
                brick_coords_df = body_brick_coords_df.merge(roi_brick_coords_df, 'inner', on=[*'zyx'])

            assert brick_coords_df.columns.tolist() == [*'zyx', 'label']
            np.save('brick-coords.npy', brick_coords_df.to_records(index=False))

        with Timer("Preparing bricks", logger):
            boxes_and_roi_bricks = []
            for coord, brick_labels in brick_coords_df.groupby([*'zyx'])['label'].agg(tuple).iteritems():
                if brick_labels == (-1,):
                    # No sparse body brick locations were found above.
                    # Search for all bodies in all bricks.
                    brick_labels = bodies

                box = np.array((coord, coord))
                box[1] += brick_shape
                box = box_intersection(box, bounding_box)

                roi_brick_box = ((box // 2**5) - roi_box_s5[0])
                roi_brick_s5 = extract_subvol(roi_vol_s5, roi_brick_box)
                boxes_and_roi_bricks.append((box, roi_brick_s5, brick_labels))

        scaled_shape = brick_shape // (2**scale)
        logger.info(f"Prepared {len(boxes_and_roi_bricks)} bricks of scale-0 shape "
                    f"{(*brick_shape[::-1],)} ({(*scaled_shape[::-1],)} at scale-{scale})")

        all_stats = []
        batches = [*iter_batches(boxes_and_roi_bricks, options["batch-size"])]
        logger.info(f"Processing {len(batches)} batches")
        for i, batch_boxes_and_bricks in enumerate(batches):
            with Timer(f"Batch {i:02d}", logger):
                batch_stats = self._execute_batch(scale, batch_boxes_and_bricks)
                all_stats.append( batch_stats )

        all_stats = pd.concat(all_stats, ignore_index=True)
        all_stats = all_stats.groupby(['body', 'roi_id'], as_index=False)['voxels'].sum()

        roi_names = pd.Series(["<none>", *options["rois"]], name='roi')
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
            """
            Args:
                box:
                    A box at scale-0
                roi_brick_s5:
                    A volume of roi voxels at scale-5, taken
                    from the region that corresponds to the box.
                    It will be upsampled to to align with the data
                    for the box.
                labels:
                    A set of labels to include in the results.
                    Other labels will be ignored.
            """
            box = box // (2**scale)
            roi_brick = upsample(roi_brick_s5, 2**(5-scale))
            assert (roi_brick.shape == box_shape(box)).all(), \
                f"{roi_brick.shape} does not match box {box.tolist()}"

            # Download seg, but erase everything except our bodies of interest.
            # Note: Service is already configured at the right scale.
            seg_brick = input_service.get_subvolume(box, scale)
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
