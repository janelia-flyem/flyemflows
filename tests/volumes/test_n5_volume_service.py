import tempfile

import pytest
import numpy as np

from neuclease.util import box_to_slicing

from flyemflows.util.n5 import export_to_multiscale_n5
from flyemflows.volumes import N5VolumeServiceReader


@pytest.fixture(scope="module")
def volume_setup():
    test_dir = tempfile.mkdtemp()
    path = f"{test_dir}/testvol.n5"
    dataset = "/some/volume-s0"
    
    config = {
        "n5": {
            "path": path,
            "dataset": dataset
        },
        "geometry": {}
    }
    
    volume = np.random.randint(100, size=(256, 256, 256), dtype=np.uint8)
    export_to_multiscale_n5(volume, path, dataset, chunks=(64,64,64), max_scale=3, downsample_method='subsample')
    return config, volume

    
def test_full_volume(volume_setup):
    config, volume = volume_setup
    reader = N5VolumeServiceReader(config)
    assert (reader.bounding_box_zyx == [(0,0,0), (256,256,256)]).all()
    full_from_n5 = reader.get_subvolume(reader.bounding_box_zyx)
    assert full_from_n5.shape == volume.shape
    assert (full_from_n5 == volume).all()


def test_slab(volume_setup):
    config, volume = volume_setup
    box = np.array([(64,0,0), (128,256,256)])
    
    slab_from_raw = volume[box_to_slicing(*box)]

    reader = N5VolumeServiceReader(config)
    slab_from_n5 = reader.get_subvolume(box)

    assert slab_from_n5.shape == slab_from_raw.shape, \
        f"Wrong shape: Expected {slab_from_raw.shape}, Got {slab_from_n5.shape}"
    assert (slab_from_n5 == slab_from_raw).all()


def test_multiscale(volume_setup):
    config, volume = volume_setup
    reader = N5VolumeServiceReader(config)
    assert (reader.bounding_box_zyx == [(0,0,0), (256,256,256)]).all()
    
    full_from_n5 = reader.get_subvolume(reader.bounding_box_zyx // 4, 2)
    
    assert (full_from_n5.shape == np.array(volume.shape) // 4).all()
    assert (full_from_n5 == volume[::4, ::4, ::4]).all()


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.volumes.test_n5_volume_service'])