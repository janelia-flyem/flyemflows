import copy
from math import pi
import pickle
import logging

import numpy as np
import pandas as pd
import dask.bag as db

from dvidutils import LabelMapper

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, round_box, SparseBlockMask, boxes_from_grid, ndindex_array
from neuclease.dvid import fetch_roi

from ..util import replace_default_entries
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService, ScaledVolumeService
from . import Workflow

logger = logging.getLogger(__name__)

COV_COLS = ['Kzz', 'Kzy', 'Kzx',
            'Kyz', 'Kyy', 'Kyx',
            'Kxz', 'Kxy', 'Kxx']


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
        boxes = _init_boxes(seg_service, options["roi"])
        logger.info(f"Processing {len(boxes)} bricks in total.")

        with Timer("Processing brick-wise stats", logger):
            # Main computation: A table for each box
            tables = (db.from_sequence(boxes, partition_size=10)
                        .map(lambda box: _process_box(seg_service, mask_service, box, options["min-size"]))
                        .compute())

            # Drop empty results
            tables = [*filter(lambda t: t is not None, tables)]

        total_rows = sum(len(t) for t in tables)
        with Timer(f"Concatenating results ({total_rows}) total rows", logger):
            # Combine stats
            full_table = pd.concat(tables, sort=True).fillna(0)
            class_cols = [*filter(lambda c: c.startswith('class'), full_table.columns)]
            full_table = full_table.astype({c: np.int32 for c in class_cols})

        with Timer("Exporting full_table.pkl", logger):
            with open('full_table.pkl', 'wb') as f:
                pickle.dump(full_table, f, protocol=pickle.HIGHEST_PROTOCOL)

        with Timer("Dropping isolated tiny mitos", logger):
            # Optimization: Immediately drop as many tiny mitos as we can.
            # (Since _process_box() now filters these out, there won't be many,
            # except for tiny objects which touch the border of a block but don't cross it.)
            nonsingletons = full_table.index.duplicated(keep=False)
            nontiny = (full_table["total_size"] >= options["min-size"])
            full_table = full_table.loc[nonsingletons | nontiny]

        with Timer("Aggregating stats across bricks", logger):
            K = full_table[COV_COLS].values.reshape((-1, 3, 3))
            μ = full_table[[*'zyx']].values.reshape((-1, 3, 1))
            μT = μ.transpose(0, 2, 1)

            # Recover the autocorrelation matrix
            R = K + μ @ μT

            Rcols = [f'R{i}{j}' for i in range(3) for j in range(3)]
            full_table = full_table.assign(**dict(zip(Rcols, R.reshape((-1, 9)).transpose())))

            # Weight each centroid and autocorrelation matrix by the object voxel count
            full_table[[*'zyx']] *= full_table[['total_size']].values
            full_table[Rcols] *= full_table[['total_size']].values

            # Aggregate by mito_id
            stats_df = full_table.groupby('mito_id')[['total_size', *class_cols, *'zyx', *Rcols]].sum()

            # Renormalize for total size
            mito_volume = stats_df[['total_size']].values
            stats_df[[*'zyx']] /= mito_volume
            stats_df[Rcols] /= mito_volume

            # Compute covariance matrix
            R = stats_df[Rcols].values.reshape((-1, 3, 3))
            μ = stats_df[[*'zyx']].values.reshape((-1, 3, 1))
            μT = μ.transpose(0, 2, 1)
            K = R - (μ @ μT)

            stats_df = stats_df.assign(**dict(zip(COV_COLS, K.reshape(-1, 9).T)))

            # Eigenvalues
            λ = np.linalg.eigvalsh(K)
            λ = λ[:, ::-1]

            # For tiny shapes or pathological cases,
            # we get tiny eigenvalues, even negative.
            # Set a floor of 0.25 to indicate a 0.5 radius (1-px diameter.)
            λ = np.maximum(0.25, λ)

            pca_radii = np.sqrt(λ)
            stats_df['pca_radius_0'] = pca_radii[:, 0]
            stats_df['pca_radius_1'] = pca_radii[:, 1]
            stats_df['pca_radius_2'] = pca_radii[:, 2]

            # The PCA radii are not scaled to be the
            # semi-major/minor radii of the ellipsoid unless we rescale them.
            # The volume of an ellipsoid with semi-radii a,b,c is 4πabc/3
            # so scale the radii up until the volume of the corresponding ellipsoid
            # matches the mito's actual volume.

            # Compute the scaling factor between the actual volume and PCA ellipsoid volume.
            pca_vol = (4/3) * np.pi * np.prod(pca_radii, axis=1)[:, None]
            s = mito_volume / pca_vol

            # Apply that factor to the PCA magnitudes to obtain better ellipsoid radii
            radii = pca_radii * np.power(s, 1/3)

            ellipsoid_vol = (4/3) * np.pi * np.prod(radii, axis=1)[:, None]
            assert np.allclose(ellipsoid_vol, mito_volume, rtol=0.01)

            stats_df['ellipsoid_radius_0'] = radii[:, 0]
            stats_df['ellipsoid_radius_1'] = radii[:, 1]
            stats_df['ellipsoid_radius_2'] = radii[:, 2]

        with Timer("Dropping remaining tiny mitos", logger):
            # Drop tiny mitos
            min_size = options["min-size"]
            min_size  # (for linter)
            stats_df = stats_df.query("total_size >= @min_size").copy()

        with Timer("Exporting stats_df.pkl", logger):
            # Integer centroids are more convenient than float
            stats_df = stats_df.astype({a: np.int32 for a in 'zyx'})

            with open('stats_df.pkl', 'wb') as f:
                pickle.dump(stats_df, f, protocol=pickle.HIGHEST_PROTOCOL)

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


def _init_boxes(volume_service, roi):
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
        if len(set(scale)) > 1:
            raise NotImplementedError("FIXME: Can't use anisotropic scaled volume with an roi")

        scale = scale[0]
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


def _process_box(seg_service, mask_service, box, min_size):
    seg_vol = seg_service.get_subvolume(box)
    if not seg_vol.any():
        # No mito components in this box
        return None

    seg_vol = _erase_tiny_interior_segments(seg_vol, min_size)
    if not seg_vol.any():
        # No mito components in this box
        return None

    mask_vol = mask_service.get_subvolume(box)
    return _object_stats(seg_vol, mask_vol, box)


def _object_stats(seg_vol, mask_vol, box):
    """
    For each mito, compute the total size, per-class sizes, centroid,
    principal axis magnitudes, and coordinate covariance matrix for
    each mito object.

    (The covariance matrix is returned in case the mito is spans multiple
    blocks and its princpal axis magnitudes need to be recomputed.)
    """
    unraveled_df = pd.DataFrame({'mito_id': seg_vol.reshape(-1),
                                 'mito_class': mask_vol.reshape(-1)})

    # Add coordinate columns to compute centroids
    # Use the narrowest dtype possible
    raster_dtype = [*filter(lambda t: np.iinfo(t).max >= box[1].max(),
                            [np.int8, np.int16, np.int32, np.int64])][0]
    raster_coords = ndindex_array(*(box[1] - box[0]), dtype=raster_dtype)
    raster_coords += box[0]
    unraveled_df = unraveled_df.assign(z=raster_coords[:, 0],
                                       y=raster_coords[:, 1],
                                       x=raster_coords[:, 2])

    # Drop non-mito-voxels
    unraveled_df = unraveled_df.iloc[(seg_vol != 0).reshape(-1)]

    # Compute the principal axes and their magnitudes (std deviations).
    # We'll also return the computed covariance matrix so we can compute
    # overall principal axes for mitos which cross from one block into another.
    cov_rows = []
    for mito_id, coords_df in unraveled_df.groupby('mito_id', sort=False)[[*'zyx']]:
        # Very important to use 64-bit float here.
        X = coords_df[[*'zyx']].values.T.astype(np.float64)
        n = len(X.T)
        μ = X.mean(axis=1)[:, None]

        # Covariance matrix.
        # It's slightly faster to compute it ourselves than to call np.cov().
        # Note:
        #   It's very important (for numerical accuracy) to center the data
        #   FIRST as done here (in the operands to @), rather than after the
        #   matrix multiplication.
        X -= μ
        K = (X @ X.T) / n

        # μz, μy, μx = μ.ravel()
        # Kzz, Kzy, Kzx, Kyz, Kyy, Kyx, Kxz, Kxy, Kxx = K.ravel()
        cov_rows.append((mito_id, n, *μ.ravel(), *K.ravel()))

    centroid_cols = ('z', 'y', 'x')
    cols = ['mito_id', 'total_size', *centroid_cols, *COV_COLS]
    stats_df = pd.DataFrame(cov_rows, columns=cols)
    stats_df = stats_df.set_index('mito_id')

    # Eigenvalues (principal axes magnitudes)
    K = stats_df[COV_COLS].values.reshape((-1, 3, 3))
    λ = np.linalg.eigvalsh(K)

    # Sort in descending order.
    λ = λ[:, ::-1]

    stats_df['λ0'] = λ[:, 0]
    stats_df['λ1'] = λ[:, 1]
    stats_df['λ2'] = λ[:, 2]

    # pivot_table() doesn't work without a data column to aggregate
    unraveled_df['voxels'] = 1

    class_df = (unraveled_df[['mito_id', 'mito_class', 'voxels']]
                .pivot_table(index='mito_id',  # noqa
                             columns='mito_class',
                             values='voxels',
                             aggfunc='sum',
                             fill_value=0))

    class_df.columns = [f"class_{c}" for c in class_df.columns]
    stats_df = class_df.merge(stats_df, 'left', left_index=True, right_index=True)
    return stats_df


def _erase_tiny_interior_segments(seg_vol, min_size):
    """
    Erase any segments that are smaller than the given
    size and don't touch the edge of the volume.
    """
    edge_mitos = (
        set(pd.unique(seg_vol[0, :, :].ravel())) |
        set(pd.unique(seg_vol[:, 0, :].ravel())) |
        set(pd.unique(seg_vol[:, :, 0].ravel())) |
        set(pd.unique(seg_vol[-1, :, :].ravel())) |
        set(pd.unique(seg_vol[:, -1, :].ravel())) |
        set(pd.unique(seg_vol[:, :, -1].ravel()))
    )

    mito_sizes = pd.Series(seg_vol.ravel()).value_counts()
    nontiny_mitos = mito_sizes[mito_sizes >= min_size].index

    keep_mitos = (edge_mitos | set(nontiny_mitos))
    keep_mitos = np.array([*keep_mitos], np.uint64)
    if len(keep_mitos) == 0:
        return np.zeros_like(seg_vol)

    # Erase everything that isn't in the keep set
    seg_vol = LabelMapper(keep_mitos, keep_mitos).apply_with_default(seg_vol, 0)
    return seg_vol
