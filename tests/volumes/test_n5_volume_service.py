import os
import copy
import tempfile

import pytest
import numpy as np

from neuclease.util import box_to_slicing, boxes_from_grid

from flyemflows.util.n5 import export_to_multiscale_n5
from flyemflows.volumes import N5VolumeService


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
    reader = N5VolumeService(config)
    assert (reader.bounding_box_zyx == [(0,0,0), (256,256,256)]).all()
    full_from_n5 = reader.get_subvolume(reader.bounding_box_zyx)
    assert full_from_n5.shape == volume.shape
    assert (full_from_n5 == volume).all()


def test_slab(volume_setup):
    config, volume = volume_setup
    box = np.array([(64,0,0), (128,256,256)])
    
    slab_from_raw = volume[box_to_slicing(*box)]

    reader = N5VolumeService(config)
    slab_from_n5 = reader.get_subvolume(box)

    assert slab_from_n5.shape == slab_from_raw.shape, \
        f"Wrong shape: Expected {slab_from_raw.shape}, Got {slab_from_n5.shape}"
    assert (slab_from_n5 == slab_from_raw).all()


def test_multiscale(volume_setup):
    config, volume = volume_setup
    reader = N5VolumeService(config)
    assert (reader.bounding_box_zyx == [(0,0,0), (256,256,256)]).all()
    
    full_from_n5 = reader.get_subvolume(reader.bounding_box_zyx // 4, 2)
    
    assert (full_from_n5.shape == np.array(volume.shape) // 4).all()
    assert (full_from_n5 == volume[::4, ::4, ::4]).all()


def test_multiscale_write(volume_setup):
    read_config, volume = volume_setup
    reader = N5VolumeService(read_config)
    
    new_path = os.path.splitext(read_config["n5"]["path"])[0] + '_WRITE_TEST.n5'

    config = copy.deepcopy(read_config)
    config["n5"]["path"] = new_path
    config["n5"]["dtype"] = str(volume.dtype)
    config["n5"]["writable"] = True
    config["geometry"]["bounding-box"] = [[0,0,0], [*volume.shape[::-1]]]
    config["geometry"]["available-scales"] = [*range(4)]
    
    writer = N5VolumeService(config)
    writer.write_subvolume(volume, (0,0,0), 0)
    
    for scale in range(1,4):
        vol = reader.get_subvolume(reader.bounding_box_zyx // 2**scale, scale)
        writer.write_subvolume(vol, (0,0,0), scale)
    
    for scale in range(4):
        box = reader.bounding_box_zyx // 2**scale
        orig_vol = reader.get_subvolume(box, scale)
        written_vol = writer.get_subvolume(box, scale)
        assert (orig_vol == written_vol).all(), f"Failed at scale {scale}"


def test_multiscale_write_nonaligned(volume_setup):
    """
    Ensure that we can write to a volume whose overall shape is not a multiple of the chunkshape.
    """
    read_config, volume = volume_setup
    reader = N5VolumeService(read_config)
    
    new_path = os.path.splitext(read_config["n5"]["path"])[0] + '_WRITE_TEST.n5'

    config = copy.deepcopy(read_config)
    config["n5"]["path"] = new_path
    config["n5"]["dtype"] = str(volume.dtype)
    config["n5"]["writable"] = True
    
    config["geometry"]["bounding-box"] = [[0,0,0], [250, 225, 200]]
    config["geometry"]["available-scales"] = [*range(4)]
    config["geometry"]["message-block-shape"] = [64,64,64]
    
    writer = N5VolumeService(config)

    # WRITE
    for scale in range(4):
        full_box = writer.bounding_box_zyx // 2**scale
        for box in boxes_from_grid(full_box, (128,128,64), clipped=True):
            vol = reader.get_subvolume(box, scale)
            writer.write_subvolume(vol, box[0], scale)

    # READ
    for scale in range(4):
        full_box = writer.bounding_box_zyx // 2**scale
        full_read = reader.get_subvolume(full_box, scale)
        full_written = writer.get_subvolume(full_box, scale)
        assert (full_read == full_written).all(), f"Failed at scale {scale}"


if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', 'tests.volumes.test_n5_volume_service']
    #args += ['-k', 'multiscale_write_nonaligned']
    pytest.main(args)
