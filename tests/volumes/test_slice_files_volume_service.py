import os
import tempfile

import pytest

import vigra
import numpy as np

from neuclease.util import box_to_slicing

from flyemflows.volumes import SliceFilesVolumeServiceReader, SliceFilesVolumeServiceWriter


@pytest.fixture(scope="module")
def read_slices_setup():
    volume = np.random.randint(100, size=(512, 256, 128), dtype=np.uint8)
    volume = vigra.taggedView(volume, 'zyx')
    
    slices_dir = tempfile.mkdtemp()
    slice_fmt = slices_dir + '/slice-{:04d}.png'
    for z, z_slice in enumerate(volume):
        vigra.impex.writeImage(z_slice, slice_fmt.format(z))

    config = {
        "slice-files": {
            "slice-path-format": slice_fmt
        },
        "geometry": {}
    }
    
    return volume, config


def test_read_full_volume(read_slices_setup):
    volume, config = read_slices_setup
    try:
        reader = SliceFilesVolumeServiceReader(config)
    except SliceFilesVolumeServiceReader.NoSlicesFoundError:
        raise RuntimeError("Test data could not be found. "
                           "It is supposed to be generated when you run the INTEGRATION TESTS. "
                           "Please run (or at least start) the integration tests first.")
    
    assert (reader.bounding_box_zyx == [(0,0,0), volume.shape]).all()
    full_from_slices = reader.get_subvolume(reader.bounding_box_zyx)
    assert full_from_slices.shape == volume.shape
    assert (full_from_slices == volume).all()


def test_read_slab(read_slices_setup):
    volume, config = read_slices_setup
    box = np.array([(0,0,0), volume.shape])
    
    # Slab from z=64 to z=128
    box[:,0] = [64, 128]
    
    slab_from_raw = volume[box_to_slicing(*box)]

    reader = SliceFilesVolumeServiceReader(config)
    slab_from_slices = reader.get_subvolume(box)

    assert slab_from_slices.shape == slab_from_raw.shape, \
        f"Wrong shape: Expected {slab_from_raw.shape}, Got {slab_from_slices.shape}"
    assert (slab_from_slices == slab_from_raw).all()


    
@pytest.fixture
def write_slices_setup():
    volume = np.random.randint(100, size=(512, 256, 128), dtype=np.uint8)
    volume = vigra.taggedView(volume, 'zyx')
    
    slices_dir = tempfile.mkdtemp()
    slice_fmt = slices_dir + '/slice-{:04d}.png'

    config = {
        "slice-files": {
            "slice-path-format": slice_fmt
        },
        "geometry": {
            "bounding-box": [[0,0,0], list(volume.shape[::-1])]
        }
    }
    
    return volume, config
    

def test_write_full_volume(write_slices_setup):
    volume, config = write_slices_setup
    writer = SliceFilesVolumeServiceWriter(config)
    writer.write_subvolume(volume, (0,0,0))
    
    reader = SliceFilesVolumeServiceReader(config)
    written_vol = reader.get_subvolume(writer.bounding_box_zyx)
    assert (written_vol == volume).all()

def test_write_slab(write_slices_setup):
    volume, config = write_slices_setup
    writer = SliceFilesVolumeServiceWriter(config)

    slab_box = writer.bounding_box_zyx.copy()
    slab_box[:,0] = [64,128]

    writer.write_subvolume(volume[64:128], (64,0,0))
    
    reader = SliceFilesVolumeServiceReader(config)
    written_vol = reader.get_subvolume(slab_box)
    assert (written_vol == volume[64:128]).all()


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.volumes.test_slice_files_volume_service'])
