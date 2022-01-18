import logging
import numpy as np
from numba import njit

from .. import VolumeServiceReader

from confiddler import flow_style

logger = logging.getLogger(__name__)

NewAxisOrderSchema = \
{
    "description": "How to present the volume, in terms of the source volume axes.",
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": { "type": "string", "enum": ["x", "y", "z", "1-x", "1-y", "1-z"] },
    "default": flow_style(["x", "y", "z"]) # no transpose
}

@njit
def numba_any(a):
    """
    Like np.any(), but uses short-circuiting to return as soon as possible.
    """
    for x in a.ravel():
        if x:
            return True
    return False


class TransposedVolumeService(VolumeServiceReader):
    """
    Wraps an existing VolumeServiceReader and presents
    a transposed or rotated view of it.
    """

    ##
    ## Example rotation constants
    ##
    #
    # (There are 24 possible orientations for a cube, so this list is incomplete.)
    #
    # Note:
    #   These constants are expressed using X,Y,Z conventions!
    #   That is how the user should enter them into the config.

    # Rotations in the XY-plane, about the Z axis
    XY_CLOCKWISE_90 = ['1-y', 'x', 'z']
    XY_COUNTERCLOCKWISE_90 = ['y', '1-x', 'z']
    XY_ROTATE_180 = ['1-x', '1-y', 'z']

    # Rotations in the XZ-plane, about the Y axis
    XZ_CLOCKWISE_90 = ['1-z', 'y', 'x']
    XZ_COUNTERCLOCKWISE_90 = ['z', 'y', '1-x']
    XZ_ROTATE_180 = ['1-x', 'y', '1-z']

    # Rotations in the YZ-plane, about the X axis
    YZ_CLOCKWISE_90 = ['x', '1-z', 'y']
    YZ_COUNTERCLOCKWISE_90 = ['x', 'z', '1-y']
    YZ_ROTATE_180 = ['x', '1-y', '1-z']

    # No-op transpose; identity
    NO_TRANSPOSE = ['x', 'y', 'z']

    # -----------------------------
    # Notes for the FlyEM hemibrain
    # -----------------------------
    #
    # The tab volumes were written in one orientation (A) when they came from the scope,
    # and then changed to a different orientation (B) when the tabs were aligned into a unified volume,
    # and then we changed to yet another orientation (C) when we loaded the data into dvid.
    #
    # The N5 volume (B) is stored here (for now, at least):
    #   /nrs/flyem/data/tmp/Z0115-22.export.n5/22-34
    #
    # Here's the rotation from the N5 orientation (B) to the final DVID orientation (C):
    FLYEM_HEMIBRAIN_ROTATE_N5_TO_DVID = ['1-z', 'x', 'y']
    #
    # And here's the inverse, i.e. to rotate from DVID (C) back to N5 (B):
    FLYEM_HEMIBRAIN_ROTATE_DVID_TO_N5 = ['y', 'z', '1-x']
    #
    # Note that it is critical to make sure the bounding-box of the dvid service
    # is pixel-perfect with respect to the bounding box of the N5 file.
    # (The block-aligned bounding-box from dvid is not acceptable.)
    # Here are the proper extents of the dvid volume, which will correspond to
    # the shape of the N5 volume after the axes are properly transposed/inverted:
    #
    #   X, Y, Z = (34427, 39725, 41394)
    #
    # Hemibrain Examples:
    #
    #   # Make the N5 volume look like the DVID volume
    #   n5_to_dvid = VolumeService.create_from_config(
    #     {'n5': {'path': '/nrs/flyem/data/tmp/Z0115-22.export.n5', 'dataset': '22-34/s0'},
    #      'adapters': {'transpose-axes': ['1-z', 'x', 'y']}})
    #
    #   # Make the DVID volume look like the N5 volume
    #   Z, Y, X = n5_to_dvid.bounding_box_zyx[1].tolist()
    #   assert (X, Y, Z) == (34427, 39725, 41394)
    #   dvid_to_n5 = VolumeService.create_from_config(
    #     {'dvid': {'server': 'emdata3:8600', 'uuid': 'a89eb', 'grayscale-name': 'grayscale'},
    #      'geometry': {'bounding-box': [[0,0,0], [X,Y,Z]]},
    #      'adapters': {'transpose-axes': ['y', 'z', '1-x']}})

    def __init__(self, original_volume_service, new_axis_order_xyz=NO_TRANSPOSE):
        """
        Note: new_axis_order_xyz should be provided in [x,y,z] order,
              exactly as it should appear in config files.
              (e.g. see NO_TRANSPOSE above).
        """
        assert len(new_axis_order_xyz) == 3
        assert not (set(new_axis_order_xyz) - set(['z', 'y', 'x', '1-z', '1-y', '1-x'])), \
            f"Invalid axis order items: {new_axis_order_xyz}"

        new_axis_order_zyx = new_axis_order_xyz[::-1]
        self.new_axis_order_zyx = new_axis_order_zyx
        self.original_volume_service = original_volume_service

        self.axis_names = [ a[-1] for a in new_axis_order_zyx ]
        assert set(self.axis_names) == set(['z', 'y', 'x'])
        self.transpose_order = tuple('zyx'.index(a) for a in self.axis_names)  # where to find the new axis in the old order
        self.rev_transpose_order = tuple(self.axis_names.index(a) for a in 'zyx')  # where to find the original axis in the new order
        self.axis_inversions = [a.startswith('1-') for a in new_axis_order_zyx]

        for i, (new, orig) in enumerate( zip(new_axis_order_zyx, 'zyx') ):
            if new != orig:
                assert self.bounding_box_zyx[0, i] == 0, \
                    "Bounding box must start at the origin for transposed axes."

        logger.info(f"Initialized TransposedVolumeService with bounding box (XYZ): {self.bounding_box_zyx[:,::-1].tolist()}")

    @property
    def base_service(self):
        return self.original_volume_service.base_service

    @property
    def dtype(self):
        return self.original_volume_service.dtype

    @property
    def block_width(self):
        return self.original_volume_service.block_width

    @property
    def preferred_message_shape(self):
        return self.original_volume_service.preferred_message_shape[(self.transpose_order,)]

    @property
    def bounding_box_zyx(self):
        return self.original_volume_service.bounding_box_zyx[:, self.transpose_order]

    @property
    def available_scales(self):
        return self.original_volume_service.available_scales

    def get_subvolume(self, new_box_zyx, scale=0):
        """
        Extract the subvolume, specified in new (transposed) coordinates from the
        original volume service, then transpose the result accordingly before returning it.
        """
        new_box_zyx = np.asarray(new_box_zyx)
        orig_box = new_box_zyx[:, self.rev_transpose_order]
        orig_bb = self.original_volume_service.bounding_box_zyx // 2**scale

        for i, inverted_name in enumerate(['1-z', '1-y', '1-x']):
            if inverted_name in self.new_axis_order_zyx:
                assert orig_bb[0, i] == 0
                Bw = _bounding_box_width = orig_bb[1, i]
                orig_box[:, i] = Bw - orig_box[:, i]  # Invert start/stop coordinates.
                orig_box[:, i] = orig_box[::-1, i]    # Now swap start/stop, since otherwise start > stop,
                                                      # and we can't call get_subvolume() with a negative step size.
                                                      # Below, we will reverse the data order for this axis after fetching it.

        inversion_slices = tuple( { False: slice(None), True: slice(None, None, -1) }[inv]
                                  for inv in self.axis_inversions )

        data = self.original_volume_service.get_subvolume(orig_box, scale)
        data = data.transpose(self.transpose_order)
        data = data[inversion_slices]

        # Force contiguous so caller doesn't have to worry about it.
        if numba_any(data):
            # Note: This is actually somewhat expensive.
            return np.asarray(data, order='C')
        else:
            # Special optimization for completely empty arrays:
            # It's much faster to just reshape the buffer instead of incurring a copy.
            return data.ravel('K').reshape(data.shape, order='C')
