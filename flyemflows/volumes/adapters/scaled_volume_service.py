import logging
from collections.abc import Collection

import numpy as np

from neuclease.util import box_to_slicing

from ...util import downsample, upsample, DOWNSAMPLE_METHODS
from .. import VolumeServiceWriter

logger = logging.getLogger(__name__)

RescaleLevelSchema = \
{
    "description": "Level to rescale the original input source when reading.\n"
                   "Presents a resized view of the original volume.\n  \n"
                   "  Examples:\n  \n"
                   "    null: No rescaling -- don't even instantiate a ScaledVolumeService adapter object.\n"
                   "          When data is requested, only the scales supported by the base service are permitted.\n"
                   "    \n"
                   "       0: No rescaling, but a ScaledVolumeService adapter is used, which allows clients to request\n"
                   "          scales that aren't natively supported by the base service.  In that case, the adapter can\n"
                   "          produce data at any scale, by downsampling the data from the base service.\n"
                   "    \n"
                   "       1: downsample by 2x, e.g. If the client requests scale 0, the adapter requests scale 1 from \n"
                   "          the base service (if available), or automatically downsamples from scale 0 data if the base\n"
                   "          service doesn't provide it natively.\n"
                   "    \n"
                   "       2: downsample by 4x, e.g. If the client requests scale 1, return (or generate) scale 3 from the base service.\n"
                   "    \n"
                   "      -1: upsample by 2x\n",
    "default": None,
    "oneOf": [
        {
            "type": "null"
        },
        {
            "type": "integer"
        },
        {
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 3,
            "maxItems": 3,
        },
        {
            "type": "object",
            "default": {},
            "properties": {
                "level": {
                    "default": 0,
                    "oneOf": [
                        {
                            "type": "integer"
                        },
                        {
                            "type": "array",
                            "items": { "type": "integer" },
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    ]
                },
                "method": {
                    "type": "string",
                    "enum": DOWNSAMPLE_METHODS,
                    "default": "subsample"
                },

                # If the source should not advertise all scales as being available,
                # you can specify a specific set of 'available-scales' here.
                # Otherwise, scales 0-9 are advertised as available.
                # (Nothing stops callers from asking for unadvertised scales, though.)
                "available-scales": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "default": [*range(10)]
                }
            }
        }
    ]
}


class ScaledVolumeService(VolumeServiceWriter):
    """
    Wraps an existing VolumeServiceReader and presents
    a scaled view of it.
    
    (Technically, this is an example of the so-called
    "decorator" GoF pattern.)
    
    Notes:
        - uint64 data is assumed to be label data, and it is downsampled accordingly, and precisely
        - All other data is assumed to be grayscale. Downsampling is IMPRECISE because this class
          does not fetch a halo before computing the downsample.
        - In both cases, upsampling is performed without any interpolation.
    """
    def __init__(self, original_volume_service, scale_delta=0, method=None, available_scales=None):
        self.original_volume_service = original_volume_service
        if not isinstance(scale_delta, Collection):
            scale_delta = 3 * [scale_delta]
        self.scale_delta = np.asarray(scale_delta[::-1])  # zyx
        self.method = method

        if available_scales is None:
            available_scales = [*range(10)]
        self._available_scales = available_scales

    @property
    def base_service(self):
        return self.original_volume_service.base_service

    @property
    def dtype(self):
        return self.original_volume_service.dtype

    @property
    def block_width(self):
        bw = int(self.original_volume_service.block_width // 2.0**self.scale_delta)
        bw = max(1, bw) # Never shrink below width of 1
        return bw

    @property
    def preferred_message_shape(self):
        # Downscale/upscale so that our message shape corresponds to the full-res message shape.
        # (We return the preferred_message_shape for OUR scale 0.)
        #
        # Note:
        #  In a previous version of this function, we wouldn't scale the original service's
        #  preferred message shape IFF our 'scale 0' happened to correspond to a 'native'\
        #  (available) scale in the original service.  The assumption was that all available
        #  scales in the original service prefer the same message shape, so there should be
        #  no reason to re-scale it.  But that has two problems:
        #    1. Many workflows use the preferred_message_shape to determine the workload
        #       blocking scheme.
        #    2. Users know that, and set the message-block-shape carefully in the config,
        #       assuming scale-0 dimensions.
        #       If we don't re-scale the preferred shape consistently for all data sources,
        #       then it's difficult for users to understand how they should write their config files.
        #
        orig = self.original_volume_service.preferred_message_shape
        ms = (orig // 2.0**self.scale_delta)
        ms = ms.astype(np.int32)

        # Never shrink below 1 pixel in each dimension
        ms = np.maximum(ms, 1)
        return ms

    @property
    def preferred_grid_offset(self):
        orig = self.original_volume_service.preferred_grid_offset
        if (orig % 2.0**self.scale_delta).any():
            raise RuntimeError("message-grid-offset is not divisible by the rescale factor")

        offset = (orig // 2.0**self.scale_delta)
        offset = offset.astype(np.int32)
        return offset

    @property
    def bounding_box_zyx(self):
        bb = (self.original_volume_service.bounding_box_zyx // 2.0**self.scale_delta).astype(np.int32)
        
        # Avoid shrinking the bounding box to 0 pixels in any dimension.
        bb[1] = np.maximum(bb[1], bb[0]+1)
        return bb

    @property
    def available_scales(self):
        return self._available_scales

    def get_subvolume(self, box_zyx, scale=0):
        """
        Extract the subvolume, specified in new (scaled) coordinates from the
        original volume service, then scale result accordingly before returning it.
        """
        if len(set(self.scale_delta)) == 1:
            return self._get_subvolume_uniform_scaling(box_zyx, scale)
        else:
            return self._get_subvolume_anisotropic_scaling(box_zyx, scale)

    def _get_subvolume_anisotropic_scaling(self, box_zyx, scale=0):
        # If some axes require downsampling and some require upsampling,
        # perform downsampling first.
        true_scale = scale + self.scale_delta
        logger.debug(f"{true_scale = }")

        orig_scales = self.original_volume_service.available_scales
        assert orig_scales == list(range(1 + max(orig_scales))), \
            "This code assumes the wrapped service has consecutive available_scales"

        orig_scales = self.original_volume_service.available_scales
        fetching_scale = max(true_scale.min(), 0)
        fetching_scale = min(fetching_scale, max(orig_scales))
        assert fetching_scale in orig_scales
        logger.debug(f"{fetching_scale = }")

        resample_scale = true_scale - fetching_scale
        resample_factor = 2.0**resample_scale
        logger.debug(f"{resample_scale = }, {resample_factor = }")

        fetching_box = rescale_box(box_zyx, 1/resample_factor)
        logger.debug(f"Fetching {fetching_box.tolist()} (ZYX) at scale {fetching_scale}")
        orig_data = self.original_volume_service.get_subvolume(fetching_box, fetching_scale)

        if (resample_scale > 0).any():
            downsample_factor = np.where(resample_scale > 0, 2**resample_scale, 1)
            logger.debug(f"Downsampling with factor {downsample_factor.tolist()}")
            downsampled_data = downsample(orig_data, downsample_factor, self.method)
        else:
            downsampled_data = orig_data

        if (resample_scale < 0).any():
            upsample_factor = np.where(resample_scale < 0, 2**-resample_scale, 1)
            logger.debug(f"Upsampling with factor {upsample_factor.tolist()}")
            upsampled_data = upsample(downsampled_data, upsample_factor)
        else:
            upsampled_data = downsampled_data

        # Note that upsampled_box_zyx is NOT necessarily the same as box_zyx,
        # due to the 'rounding out' that may have been necessary when creating orig_box_zyx.
        # Therefore we have to crop out the part we need.
        upsampled_box_zyx = rescale_box(fetching_box, resample_factor)
        relative_box = box_zyx - upsampled_box_zyx[0]
        requested_data = upsampled_data[box_to_slicing(*relative_box)]

        # Force contiguous so caller doesn't have to worry about it.
        return np.asarray(requested_data, order='C')

    def _get_subvolume_uniform_scaling(self, box_zyx, scale):
        box_zyx = np.asarray(box_zyx)
        true_scale = scale + self.scale_delta[0]

        if true_scale in self.original_volume_service.available_scales:
            # The original source already has the data at the necessary scale.
            return self.original_volume_service.get_subvolume( box_zyx, true_scale )

        # Start with the closest scale we've got
        base_scales = np.array(self.original_volume_service.available_scales)
        i_best = np.abs(base_scales - true_scale).argmin()
        best_base_scale = base_scales[i_best]

        delta_from_best = true_scale - best_base_scale

        if delta_from_best > 0:
            orig_box_zyx = box_zyx * 2**delta_from_best
            orig_data = self.original_volume_service.get_subvolume(orig_box_zyx, best_base_scale)

            if self.method:
                downsampled_data = downsample( orig_data, 2**delta_from_best, self.method )
            elif np.dtype(self.dtype) == np.uint64:
                # Assume that uint64 means labels.

                ## FIXME: Our C++ method for downsampling ('labels')
                ##        seems to have a bad build at the moment (it segfaults and/or produces zeros)
                ##        For now, we use the 'labels-numba' method
                downsampled_data = downsample( orig_data, 2**delta_from_best, 'labels-numba' )
            else:
                downsampled_data = downsample( orig_data, 2**delta_from_best, 'block-mean' )
            return downsampled_data
        else:
            upsample_factor = int(2**-delta_from_best)
            orig_box_zyx = rescale_box(box_zyx, np.array(3*(upsample_factor,)))
            orig_data = self.original_volume_service.get_subvolume(orig_box_zyx, best_base_scale)

            upsampled_data = upsample(orig_data, upsample_factor)
            relative_box = box_zyx - upsample_factor*orig_box_zyx[0]
            requested_data = upsampled_data[box_to_slicing(*relative_box)]

            # Force contiguous so caller doesn't have to worry about it.
            return np.asarray(requested_data, order='C')


    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        """
        Write the given data into the original volume source
        at the given scale, but upsample/downsample it first
        according to our scale_delta.
        
        The scale_delta indicates how much downsampling to apply
        to data that is read, and how much upsampling to apply
        to data that is written.
        
        But in general, workflows should only be writing into
        scale 0 if they are using a scaled volume service.
        Other pyramid levels should be computed afterwards.
        """
        if len(set(self.scale_delta)) > 1:
            raise NotImplementedError("Anisotropic rescaling not supported by this method.")

        scale_delta = int(self.scale_delta[0])

        offset_zyx = np.asarray(offset_zyx)
        offset_zyx = (offset_zyx * 2.0**scale_delta).astype(int)
        
        if scale_delta >= 0:
            upsampled_data = upsample(subvolume, 2**scale_delta)
            self.original_volume_service.write_subvolume(upsampled_data, offset_zyx, scale)
        else:
            if np.dtype(self.dtype) in (np.uint64, np.uint32):
                # Assume that uint64/uint32 means labels.
                downsampled_data = downsample( subvolume, 2**(-scale_delta), 'subsample' )
            else:
                downsampled_data = downsample( subvolume, 2**(-scale_delta), 'block-mean' )
            self.original_volume_service.write_subvolume(downsampled_data, offset_zyx, scale)
            

    def sample_labels(self, points_zyx, scale=0, npartitions=1024):
        if len(set(self.scale_delta)) > 1:
            raise NotImplementedError("Anisotropic rescaling not supported by this method.")
        scale_delta = self.scale_delta[0]

        true_scale = scale + scale_delta
        if true_scale in self.original_volume_service.available_scales:
            # The original source already has the data at the necessary scale.
            return self.original_volume_service.sample_labels(points_zyx, true_scale, npartitions)

        # FIXME: It would be good to select the "best scale" as is done in get_subvolume() above.
        return super().sample_labels(points_zyx, scale, npartitions)


    def sparse_brick_coords_for_labels(self, labels, clip=True):
        if len(set(self.scale_delta)) > 1:
            raise NotImplementedError("Anisotropic rescaling not supported by this method.")
        scale_delta = int(self.scale_delta[0])

        coords_df = self.original_volume_service.sparse_brick_coords_for_labels(labels, clip)
        coords_df[[*'zyx']] //= (2.0**scale_delta)
        coords_df[[*'zyx']] = coords_df[[*'zyx']].astype(np.int32)
        return coords_df


def rescale_box(box, downsample_factor):
    """
    Given a box (i.e. start and stop coordinates) and a
    block_shape (downsampling factor), return the corresponding box
    in downsampled coordinates, "rounding out" if the box is not an
    even multiple of the block shape.

    Fractional downsample_factor is also permitted, which
    implies upsampling (and the box gets bigger).
    """
    box = np.asarray(box)
    downsample_factor = np.asarray(downsample_factor)
    assert downsample_factor.shape[0] == box.shape[1]
    rescaled_box = np.zeros_like(box)
    rescaled_box[0] = np.floor(box[0] / downsample_factor)
    rescaled_box[1] = np.ceil(box[1] / downsample_factor)

    if np.issubdtype(box.dtype, np.integer):
        rescaled_box = np.round(rescaled_box).astype(box.dtype)

    return rescaled_box
