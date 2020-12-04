import numpy as np

from neuclease.util import box_to_slicing

from ...util import downsample, upsample, DOWNSAMPLE_METHODS
from .. import VolumeServiceWriter

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
        {"type": "integer"},
        {"type": "null"},
        {
            "type": "object",
            "default": {},
            "properties": {
                "level": {"type": "integer", "default": 0},

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
        self.scale_delta = scale_delta
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
        bw = int(self.original_volume_service.block_width // 2.**self.scale_delta)
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
        ms = (orig // 2**self.scale_delta)
        ms = ms.astype(np.int32)
        ms = np.maximum(ms, 1) # Never shrink below 1 pixel in each dimension
        return ms

    @property
    def bounding_box_zyx(self):
        bb = (self.original_volume_service.bounding_box_zyx // 2**self.scale_delta).astype(np.int32)
        
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
        box_zyx = np.asarray(box_zyx)
        true_scale = scale + self.scale_delta
        
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
            orig_box_zyx = downsample_box(box_zyx, np.array(3*(upsample_factor,)))
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
        offset_zyx = np.asarray(offset_zyx)
        offset_zyx = (offset_zyx * 2**self.scale_delta).astype(int)
        
        if self.scale_delta >= 0:
            upsampled_data = upsample(subvolume, 2**(self.scale_delta))
            self.original_volume_service.write_subvolume(upsampled_data, offset_zyx, scale)
        else:
            if np.dtype(self.dtype) == np.uint64:
                # Assume that uint64 means labels.
                
                ## FIXME: Our C++ method for downsampling ('labels')
                ##        seems to have a bad build at the moment (it segfaults and/or produces zeros)
                ##        For now, we use the 'labels-numba' method
                downsampled_data = downsample( subvolume, 2**(-self.scale_delta), 'labels-numba' )
            else:
                downsampled_data = downsample( subvolume, 2**(-self.scale_delta), 'block-mean' )
            self.original_volume_service.write_subvolume(downsampled_data, offset_zyx, scale)
            

    def sample_labels(self, points_zyx, scale=0, npartitions=1024):
        true_scale = scale + self.scale_delta

        if true_scale in self.original_volume_service.available_scales:
            # The original source already has the data at the necessary scale.
            return self.original_volume_service.sample_labels(points_zyx, true_scale, npartitions)

        # FIXME: It would be good to select the "best scale" as is done in get_subvolume() above.
        return super().sample_labels(points_zyx, scale, npartitions)


    def sparse_brick_coords_for_labels(self, labels, clip=True):
        coords_df = self.original_volume_service.sparse_brick_coords_for_labels(labels, clip)
        coords_df[['z', 'y', 'x']] //= (2**self.scale_delta)
        return coords_df


def downsample_box( box, block_shape ):
    """
    Given a box (i.e. start and stop coordinates) and a
    block_shape (downsampling factor), return the corresponding box
    in downsampled coordinates, "rounding out" if the box is not an
    even multiple of the block shape.
    """
    assert block_shape.shape[0] == box.shape[1]
    downsampled_box = np.zeros_like(box)
    downsampled_box[0] = box[0] // block_shape
    downsampled_box[1] = (box[1] + block_shape - 1) // block_shape
    return downsampled_box
