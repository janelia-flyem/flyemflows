import numpy as np

from neuclease.util import box_to_slicing, SparseBlockMask

from ...util import downsample, upsample
from .. import VolumeServiceReader

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
                   "          produced data at any scale, by downsampling the data from the base service.\n"
                   "    \n"
                   "       1: downsample by 2x, e.g. If the client requests scale 0, the adapter requests scale 1 from \n"
                   "          the base service (if available), or automatically downsamples from scale 0 data if the base\n"
                   "          service doesn't provide it natively.\n"
                   "    \n"
                   "       2: downsample by 4x, e.g. If the client requests scale 1, return (or generate) scale 3 from the base service.\n"
                   "    \n"
                   "      -1: upsample by 2x\n",
    "oneOf": [ {"type": "integer"}, {"type": "null"} ],
    "default": None
}

class ScaledVolumeService(VolumeServiceReader):
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
    def __init__(self, original_volume_service, scale_delta=0):
        self.original_volume_service = original_volume_service
        self.scale_delta = scale_delta

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
        # We return the preferred_message_shape for scale 0.
        # If we're able to make a direct call to the original service (i.e. we don't have to downample it ourselves),
        # then just return the native preferred_message_shape
        #
        # This is a bit ugly.  See TODO in get_subvolume().
        if self.scale_delta in self.original_volume_service.available_scales:
            return self.original_volume_service.preferred_message_shape
        else:
            ms = (self.original_volume_service.preferred_message_shape // 2**self.scale_delta).astype(np.uint32)
            ms = np.maximum(ms, 1) # Never shrink below 1 pixel in each dimension
            return ms

    @property
    def bounding_box_zyx(self):
        bb = (self.original_volume_service.bounding_box_zyx // 2**self.scale_delta).astype(np.uint32)
        
        # Avoid shrinking the bounding box to 0 pixels in any dimension.
        bb[1] = np.maximum(bb[1], bb[0]+1)
        return bb

    @property
    def available_scales(self):
        return list(range(10)) # arbitrary limit

    def get_subvolume(self, box_zyx, scale=0):
        """
        Extract the subvolume, specified in new (scaled) coordinates from the
        original volume service, then scale result accordingly before returning it.
        """
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

            if self.dtype == np.uint64:
                # Assume that uint64 means labels.
                
                ## FIXME: Our C++ method for downsampling ('labels')
                ##        seems to have a bad build at the moment (it segfaults and/or produces zeros)
                ##        For now, we use the 'labels-numba' method
                downsampled_data = downsample( orig_data, 2**delta_from_best, 'labels-numba' )
            else:
                downsampled_data = downsample( orig_data, 2**delta_from_best, 'grayscale' )
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
