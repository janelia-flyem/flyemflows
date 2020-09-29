import os
import copy
import pickle
import logging

import numpy as np
import pandas as pd
import dask.bag as db

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, round_box, SparseBlockMask, boxes_from_grid, iter_batches, ndindex_array
from neuclease.dvid import fetch_labelmap_voxels, fetch_roi

from ..util import replace_default_entries
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService, ScaledVolumeService
from . import Workflow

logger = logging.getLogger(__name__)


class MitoStats(Workflow):
    """
    Given a mitochondria segmentation and a mito "mask" segmentation,
    compute the centroid of each mito and the number of voxels it
    contains of each mask class.

    Note:
        For mito objects that are not convex, the computed centroid
        will not necessarily fall within the mito object itself.

        See the following post-processing script, which can be used
        to "correct" the centroids by moving them to a point within
        the actual object:

            flyemflows/workflow/util/correct_centroids.py
    """
    MitoStatsOptionsSchema = \
    {
        "type": "object",
        "description": "Settings specific to the MitoStats workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "min-size": {
                "description": "Don't include stats for mitochondria smaller than this (in voxels)\n",
                "type": "number",
                "default": 10e3
            },
            "roi": {
                "description": "Limit analysis to bricks that intersect the given DVID ROI.\n",
                "type": "object",
                "default": {},
                "properties": {
                    "server": {
                        "description": "dvid server for the ROI. If not provided, the input server will be used (if possible).",
                        "type": "string",
                        "default": ""
                    },
                    "uuid": {
                        "description": "dvid UUID for the ROI.  If not provided, the input UUID will be used (if possible).",
                        "type": "string",
                        "default": ""
                    },
                    "name": {
                        "description": "name of the ROI",
                        "type": "string",
                        "default": ""
                    },
                    "scale": {
                        "description": "Optionally rescale the ROI.\n"
                                       "Scale 0 means each ROI voxel is 32px wide in full-res coordinates.\n"
                                       "Scale 1 means 16px, etc.  By default, choose the scale automatically by inspecting the input rescale-level.\n",
                        "default": None,
                        "oneOf": [
                            {"type": "null"},
                            {"type": "integer"}
                        ]
                    }
                }
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "mito-seg": SegmentationVolumeSchema,
        "mito-masks": SegmentationVolumeSchema,
        "mitostats": MitoStatsOptionsSchema
    })

    @classmethod
    def schema(cls):
        return MitoStats.Schema

    def execute(self):
        options = self.config["mitostats"]
        seg_service, mask_service = self.init_services()

        # Boxes are determined by the left volume/labels/roi
        boxes = self.init_boxes( seg_service, options["roi"] )
        logger.info(f"Processing {len(boxes)} bricks in total.")

        with Timer("Processing brick-wise stats", logger):
            # Main computation: A table for each box
            tables = (db.from_sequence(boxes, partition_size=10)
                        .map(lambda box: _process_box(seg_service, mask_service, box))
                        .compute())

            # Drop empty results
            tables = [*filter(lambda t: t is not None, tables)]

        total_rows = sum(len(t) for t in tables)
        with Timer(f"Concatenating results ({total_rows}) total rows", logger):
            # Combine stats
            full_table = pd.concat(tables, sort=True).fillna(0)
            class_cols = [*filter(lambda c: c.startswith('class'), full_table.columns)]
            full_table = full_table.astype({c: np.int32 for c in class_cols})

        with Timer(f"Exporting full_table.pkl", logger):
            with open('full_table.pkl', 'wb') as f:
                pickle.dump(full_table, f, protocol=pickle.HIGHEST_PROTOCOL)

        with Timer("Dropping isolated tiny mitos", logger):
            # Optimization: Immediately drop as many tiny mitos as we can
            nonsingletons = full_table.index.duplicated(keep=False)
            nontiny = (full_table["total_size"] >= options["min-size"])
            full_table = full_table.loc[nonsingletons | nontiny]

        with Timer("Aggregating stats across bricks", logger):
            # Weight each brick centroid by the brick's voxel count before taking the mean
            full_table[[*'zyx']] *= full_table[['total_size']].values
            stats_df = full_table.groupby('mito_id').sum()
            stats_df[[*'zyx']] /= stats_df[['total_size']].values

        with Timer("Dropping remaining tiny mitos", logger):
            # Drop tiny mitos
            min_size = options["min-size"]
            stats_df = stats_df.query("total_size >= @min_size").copy()

        with Timer(f"Exporting stats_df.pkl", logger):
            # Assume all centroids are 'exact' by default (overwritten below if necessary)
            stats_df['centroid_type'] = 'exact'

            stats_df = stats_df.astype({a: np.int32 for a in 'zyx'})
            stats_df = stats_df[[*'xyz', 'total_size', *class_cols, 'centroid_type']]

            with open('stats_df.pkl', 'wb') as f:
                pickle.dump(stats_df, f, protocol=pickle.HIGHEST_PROTOCOL)

        # TODO:
        # - Check centroid validity (at scale 0?)
        # - Determine ROI for each centroid

    def init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        """
        mgr_config = self.config["resource-manager"]
        seg_config = self.config["mito-seg"]
        mask_config = self.config["mito-masks"]

        resource_mgr_client = ResourceManagerClient( mgr_config["server"], mgr_config["port"] )
        seg_service = VolumeService.create_from_config( seg_config, resource_mgr_client )
        logger.info(f"Bounding box: {seg_service.bounding_box_zyx[:,::-1].tolist()}")

        replace_default_entries(mask_config["geometry"]["bounding-box"], seg_service.bounding_box_zyx[:, ::-1])
        mask_service = VolumeService.create_from_config( mask_config, resource_mgr_client )

        if (seg_service.preferred_message_shape != mask_service.preferred_message_shape).any():
            raise RuntimeError("Your input volume and mask volume must use the same message-block-shape.")

        return seg_service, mask_service

    def init_boxes(self, volume_service, roi):
        if not roi["name"]:
            boxes = boxes_from_grid(volume_service.bounding_box_zyx,
                                    volume_service.preferred_message_shape,
                                    clipped=True)
            return np.array([*boxes])

        base_service = volume_service.base_service

        if not roi["server"] or not roi["uuid"]:
            assert isinstance(base_service, DvidVolumeService), \
                "Since you aren't using a DVID input source, you must specify the ROI server and uuid."

        roi["server"] = (roi["server"] or volume_service.server)
        roi["uuid"] = (roi["uuid"] or volume_service.uuid)

        if roi["scale"] is not None:
            scale = roi["scale"]
        elif isinstance(volume_service, ScaledVolumeService):
            scale = volume_service.scale_delta
            assert scale <= 5, \
                "The 'roi' option doesn't support volumes downscaled beyond level 5"
        else:
            scale = 0

        brick_shape = volume_service.preferred_message_shape
        assert not (brick_shape % 2**(5-scale)).any(), \
            "If using an ROI, select a brick shape that is divisible by 32"

        seg_box = volume_service.bounding_box_zyx
        seg_box = round_box(seg_box, 2**(5-scale))
        seg_box_s0 = seg_box * 2**scale
        seg_box_s5 = seg_box // 2**(5-scale)

        with Timer(f"Fetching mask for ROI '{roi['name']}' ({seg_box_s0[:, ::-1].tolist()})", logger):
            roi_mask_s5, _ = fetch_roi(roi["server"], roi["uuid"], roi["name"], format='mask', mask_box=seg_box_s5)

        # SBM 'full-res' corresponds to the input service voxels, not necessarily scale-0.
        sbm = SparseBlockMask(roi_mask_s5, seg_box, 2**(5-scale))
        boxes = sbm.sparse_boxes(brick_shape)

        # Clip boxes to the true (not rounded) bounding box
        boxes[:, 0] = np.maximum(boxes[:, 0], volume_service.bounding_box_zyx[0])
        boxes[:, 1] = np.minimum(boxes[:, 1], volume_service.bounding_box_zyx[1])
        return boxes


def _process_box(seg_service, mask_service, box):
    # TODO: I could filter out tiny isolated mitos here, before sending the results back...
    seg_vol = seg_service.get_subvolume(box)
    if not seg_vol.any():
        # No mito components in this box
        return None

    mask_vol = mask_service.get_subvolume(box)

    unraveled_df = pd.DataFrame({'mito_id': seg_vol.reshape(-1),
                                 'mito_class': mask_vol.reshape(-1)})

    # pivot_table() doesn't work without a data column to aggregate
    unraveled_df['voxels'] = 1

    # Add coordinate columns to compute centroids
    # Use the narrowest dtype possible
    raster_dtype = [*filter(lambda t: np.iinfo(t).max >= box[1].max(), [np.int8, np.int16, np.int32, np.int64])][0]
    raster_coords = ndindex_array(*(box[1] - box[0]), dtype=raster_dtype)
    raster_coords += box[0]
    unraveled_df['z'] = raster_dtype(0)
    unraveled_df['y'] = raster_dtype(0)
    unraveled_df['x'] = raster_dtype(0)
    unraveled_df[['z', 'y', 'x']] = raster_coords

    # Drop non-mito-voxels
    unraveled_df = unraveled_df.iloc[(seg_vol != 0).reshape(-1)]

    table = (unraveled_df[['mito_id', 'mito_class', 'voxels']]
                .pivot_table(index='mito_id',  # noqa
                            columns='mito_class',
                            values='voxels',
                            aggfunc='sum',
                            fill_value=0))

    table.columns = [f"class_{c}" for c in table.columns]
    table['total_size'] = table.sum(axis=1).astype(np.int32)

    # Compute box-local centroid for each mito
    mito_points = (unraveled_df[[*'zyx', 'mito_id']]
                    .astype({'z': np.float64, 'y': np.float64, 'x': np.float64})
                    .groupby('mito_id').mean())
    table = table.merge(mito_points, 'left', left_index=True, right_index=True)
    return table
