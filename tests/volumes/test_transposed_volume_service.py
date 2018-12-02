import tempfile
import unittest

import h5py
import numpy as np

from neuclease.util import box_to_slicing
from flyemflows.volumes import Hdf5VolumeService, TransposedVolumeService
from flyemflows.volumes.scaled_volume_service import ScaledVolumeService

class TestTransposedVolumeService(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        test_dir = tempfile.mkdtemp()
        test_file = f'{test_dir}/scaled-volume-test.h5'
        
        full_volume = np.random.randint(255, size=(256,256,256))
        with h5py.File(test_file, 'w') as f:
            f['volume'] = full_volume

        box_zyx = np.array([[0,0,0], [100,200,256]])
        box_xyz = box_zyx[:,::-1]

        cls.RAW_VOLUME_DATA = full_volume[box_to_slicing(*box_zyx)]
        
        cls.VOLUME_CONFIG = {
          "hdf5": {
            "path": test_file,
            "dataset": "volume"
          },
          "geometry": {
              "bounding-box": box_xyz.tolist(),
              "available-scales": [0] # Ensure only the first scale is used.
          }
        }
        
        # First, hdf5 alone
        h5_reader = Hdf5VolumeService(cls.VOLUME_CONFIG)
        assert (h5_reader.bounding_box_zyx == box_zyx).all()
        full_from_h5 = h5_reader.get_subvolume(h5_reader.bounding_box_zyx)
        assert full_from_h5.shape == (*(box_zyx[1] - box_zyx[0]),)
        assert (full_from_h5 == cls.RAW_VOLUME_DATA).all()
    
        cls.h5_reader = h5_reader
        cls.full_from_h5 = full_from_h5


    def test_full_volume(self):
        # First, h5 alone
        h5_reader = Hdf5VolumeService(self.VOLUME_CONFIG)
        assert (h5_reader.bounding_box_zyx == [(0,0,0), (100,200,256)]).all()
        full_from_h5 = h5_reader.get_subvolume(h5_reader.bounding_box_zyx)
        assert full_from_h5.shape == self.RAW_VOLUME_DATA.shape
        assert (full_from_h5 == self.RAW_VOLUME_DATA).all()

        # Check API
        transposed_reader = TransposedVolumeService(h5_reader, ['x', 'y', 'z'])
        assert transposed_reader.base_service == h5_reader
        assert len(transposed_reader.service_chain) == 2
        assert transposed_reader.service_chain[0] == transposed_reader
        assert transposed_reader.service_chain[1] == h5_reader
        
        # Now use transposed reader, but with identity transposition
        transposed_reader = TransposedVolumeService(h5_reader, ['x', 'y', 'z'])
        assert (transposed_reader.bounding_box_zyx == h5_reader.bounding_box_zyx).all()
        assert (transposed_reader.preferred_message_shape == h5_reader.preferred_message_shape).all()
        assert transposed_reader.block_width == h5_reader.block_width
        assert transposed_reader.dtype == h5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_h5).all()
        assert full_transposed.flags.c_contiguous

        # Now transpose x and y (reflect across diagonal line at y=x)
        transposed_reader = TransposedVolumeService(h5_reader, ['y', 'x', 'z'])
        assert (transposed_reader.bounding_box_zyx == h5_reader.bounding_box_zyx[:, (0,2,1)]).all()
        assert (transposed_reader.preferred_message_shape == h5_reader.preferred_message_shape[((0,2,1),)]).all()
        assert transposed_reader.block_width == h5_reader.block_width
        assert transposed_reader.dtype == h5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_h5.transpose(0, 2, 1)).all()
        assert full_transposed.flags.c_contiguous

        # Invert x and y (but don't transpose)
        # Equivalent to 180 degree rotation
        transposed_reader = TransposedVolumeService(h5_reader, ['1-x', '1-y', 'z'])
        assert (transposed_reader.bounding_box_zyx == h5_reader.bounding_box_zyx).all()
        assert (transposed_reader.preferred_message_shape == h5_reader.preferred_message_shape).all()
        assert transposed_reader.block_width == h5_reader.block_width
        assert transposed_reader.dtype == h5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_h5[:,::-1, ::-1]).all()
        assert full_transposed.flags.c_contiguous

        # XY 90 degree rotation, clockwise about the Z axis
        transposed_reader = TransposedVolumeService(h5_reader, TransposedVolumeService.XY_CLOCKWISE_90)
        assert (transposed_reader.bounding_box_zyx == h5_reader.bounding_box_zyx[:, (0,2,1)]).all()
        assert (transposed_reader.preferred_message_shape == h5_reader.preferred_message_shape[((0,2,1),)]).all()
        assert transposed_reader.block_width == h5_reader.block_width
        assert transposed_reader.dtype == h5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_h5[:, ::-1, :].transpose(0,2,1)).all()
        assert full_transposed.flags.c_contiguous
        
        # Check the corners of the first plane: should be rotated clockwise
        z_slice_h5 = full_from_h5[0]
        z_slice_transposed = full_transposed[0]
        assert z_slice_h5[0,0] == z_slice_transposed[0,-1]
        assert z_slice_h5[0,-1] == z_slice_transposed[-1,-1]
        assert z_slice_h5[-1,-1] == z_slice_transposed[-1,0]
        assert z_slice_h5[-1,0] == z_slice_transposed[0,0]

        # Verify that subvolume requests work correctly
        box = [[10,20,30], [20, 40, 60]]
        subvol_transposed = transposed_reader.get_subvolume(box)
        assert (subvol_transposed == full_transposed[box_to_slicing(*box)]).all()
        assert subvol_transposed.flags.c_contiguous

        # XZ degree rotation, clockwise about the Y axis
        transposed_reader = TransposedVolumeService(h5_reader, TransposedVolumeService.XZ_CLOCKWISE_90)
        assert (transposed_reader.bounding_box_zyx == h5_reader.bounding_box_zyx[:, (2,1,0)]).all()
        assert (transposed_reader.preferred_message_shape == h5_reader.preferred_message_shape[((2,1,0),)]).all()
        assert transposed_reader.block_width == h5_reader.block_width
        assert transposed_reader.dtype == h5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_h5[::-1, :, :].transpose(2,1,0)).all()
        assert full_transposed.flags.c_contiguous
        
        # Check the corners of the first plane: should be rotated clockwise
        y_slice_h5 = full_from_h5[:, 0, :]
        y_slice_transposed = full_transposed[:, 0, :]
        assert y_slice_h5[0,0] == y_slice_transposed[0,-1]
        assert y_slice_h5[0,-1] == y_slice_transposed[-1,-1]
        assert y_slice_h5[-1,-1] == y_slice_transposed[-1,0]
        assert y_slice_h5[-1,0] == y_slice_transposed[0,0]

        # YZ degree rotation, clockwise about the X axis
        transposed_reader = TransposedVolumeService(h5_reader, TransposedVolumeService.YZ_CLOCKWISE_90)
        assert (transposed_reader.bounding_box_zyx == h5_reader.bounding_box_zyx[:, (1,0,2)]).all()
        assert (transposed_reader.preferred_message_shape == h5_reader.preferred_message_shape[((1,0,2),)]).all()
        assert transposed_reader.block_width == h5_reader.block_width
        assert transposed_reader.dtype == h5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_h5[::-1, :, :].transpose(1,0,2)).all()
        assert full_transposed.flags.c_contiguous
        
        # Check the corners of the first plane: should be rotated clockwise
        x_slice_h5 = full_from_h5[:, :, 0]
        x_slice_transposed = full_transposed[:, :, 0]
        assert x_slice_h5[0,0] == x_slice_transposed[0,-1]
        assert x_slice_h5[0,-1] == x_slice_transposed[-1,-1]
        assert x_slice_h5[-1,-1] == x_slice_transposed[-1,0]
        assert x_slice_h5[-1,0] == x_slice_transposed[0,0]

        # Multiple rotations (the hemibrain h5 -> DVID transform)
        transposed_reader = TransposedVolumeService(h5_reader, ['1-z', 'x', 'y'])
        assert (transposed_reader.bounding_box_zyx == h5_reader.bounding_box_zyx[:, (1,2,0)]).all()
        assert (transposed_reader.preferred_message_shape == h5_reader.preferred_message_shape[((1,2,0),)]).all()
        assert transposed_reader.block_width == h5_reader.block_width
        assert transposed_reader.dtype == h5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_h5[::-1, :, :].transpose(1,2,0)).all()
        assert full_transposed.flags.c_contiguous


    def test_multiscale(self):
        SCALE = 2
        h5_reader = Hdf5VolumeService(self.VOLUME_CONFIG)
        h5_reader = ScaledVolumeService(h5_reader, 0)

        # No transpose
        transposed_reader = TransposedVolumeService(h5_reader, TransposedVolumeService.NO_TRANSPOSE)
        from_h5 = h5_reader.get_subvolume(h5_reader.bounding_box_zyx // 2**SCALE, SCALE)
        from_transposed = transposed_reader.get_subvolume(h5_reader.bounding_box_zyx // 2**SCALE, SCALE)
        assert (from_transposed == from_h5).all() 

        # XZ degree rotation, clockwise about the Y axis
        transposed_reader = TransposedVolumeService(h5_reader, TransposedVolumeService.XY_CLOCKWISE_90)
        from_h5 = h5_reader.get_subvolume(h5_reader.bounding_box_zyx // 2**SCALE, SCALE)
        from_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx // 2**SCALE, SCALE)
        assert (from_transposed == from_h5[:,::-1,:].transpose(0,2,1)).all() 


if __name__ == "__main__":
    unittest.main()
