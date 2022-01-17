import os
import tempfile
import pytest

import zarr
import numpy as np

from neuclease.util import box_to_slicing

from flyemflows.volumes import ZarrVolumeService


@pytest.fixture()
def volume_setup():
    tmpdir = tempfile.mkdtemp()
    path = f"{tmpdir}/test_zarr_service_testvol.zarr"
    dataset = "/some/volume"

    config = {
        "zarr": {
            "path": path,
            "dataset": dataset,
            "global-offset": [1000,2000,3000]
        },
        "geometry": {}
    }

    volume = np.random.randint(100, size=(512, 256, 128))

    store = zarr.NestedDirectoryStore(path)
    f = zarr.open(store=store, mode='w')
    f.create_dataset(dataset, data=volume, chunks=(64,64,64), compressor=None)

    return config, volume


def test_read(volume_setup):
    config, volume = volume_setup
    global_offset = config["zarr"]["global-offset"][::-1]

    service = ZarrVolumeService(config)
    assert (service.bounding_box_zyx - global_offset == [(0,0,0), volume.shape]).all()
    assert service.dtype == volume.dtype

    # Service INSERTS geometry into config if necessary
    assert (config["geometry"]["bounding-box"] == service.bounding_box_zyx[:, ::-1]).all()

    box = np.array([(30,40,50), (50,60,70)])
    subvol = service.get_subvolume(box + global_offset)
    assert (subvol == volume[box_to_slicing(*box)]).all()

    # Check out-of-bounds read (should be zeros)
    oob_box = box.copy()
    oob_box[1,2] = 500
    subvol = service.get_subvolume(oob_box + global_offset)

    # In-bounds portion should match
    assert (subvol[box_to_slicing(*box - box[0])] == volume[box_to_slicing(*box)]).all()

    # Everything else should be zeros
    assert (subvol[:, :, 128:] == 0).all()

    #
    # Check sample_labels()
    #
    points = [np.random.randint(d, size=(10,)) for d in volume.shape]
    points = np.transpose(points)
    global_points = points + global_offset
    labels = service.sample_labels(global_points)
    assert (labels == volume[(*points.transpose(),)]).all()


def test_write(volume_setup):
    tmpdir = tempfile.mkdtemp()
    config, volume = volume_setup
    global_offset = config["zarr"]["global-offset"][::-1]

    config["zarr"]["path"] = f"{tmpdir}/test_zarr_service_testvol_WRITE.zarr"
    if os.path.exists(config["zarr"]["path"]):
        os.unlink(config["zarr"]["path"])

    # Can't initialize service if file doesn't exist
    with pytest.raises(RuntimeError) as excinfo:
        ZarrVolumeService(config)
        assert 'create-if-necessary' in str(excinfo.value)

    assert not os.path.exists(config["zarr"]["path"])
    config["zarr"]["create-if-necessary"] = True
    config["zarr"]["creation-settings"] = {
        "shape": [*volume.shape][::-1],
        "dtype": str(volume.dtype),
        "chunk-shape": [32,32,32],
        "max-scale": 0
    }

    # Write some data
    box = [(30,40,50), (50,60,70)]
    box = np.array(box)
    subvol = volume[box_to_slicing(*box)]
    service = ZarrVolumeService(config)
    service.write_subvolume(subvol, box[0] + global_offset)

    # Read it back.
    subvol = service.get_subvolume(box + global_offset)
    assert (subvol == volume[box_to_slicing(*box)]).all()

    # Write some out-of-bounds zeros
    oob_box = box.copy()
    oob_box[1,2] = 500
    subvol = np.zeros(oob_box[1] - oob_box[0], int)
    service.write_subvolume(subvol, oob_box[0] + global_offset)

    # Read it back.
    readback = service.get_subvolume(oob_box + global_offset)
    assert (readback == subvol).all()

    # Try writing something other than zeros -- should fail
    subvol[:,:,-1] = 1
    with pytest.raises(RuntimeError):
        service.write_subvolume(subvol, oob_box[0] + global_offset)


def test_write_empty_blocks(volume_setup):
    tmpdir = tempfile.mkdtemp()
    config, volume = volume_setup
    del config["zarr"]["global-offset"]

    config["zarr"]["path"] = f"{tmpdir}/test_zarr_service_testvol_WRITE_EMPTY_BLOCKS.zarr"
    if os.path.exists(config["zarr"]["path"]):
        os.unlink(config["zarr"]["path"])

    assert not os.path.exists(config["zarr"]["path"])
    config["zarr"]["write-empty-blocks"] = False
    config["zarr"]["create-if-necessary"] = True
    config["zarr"]["creation-settings"] = {
        "shape": [*volume.shape][::-1],
        "dtype": str(volume.dtype),
        "chunk-shape": [32,32,32],
        "max-scale": 0
    }

    # Write some data
    box = np.array([(0,0,0), (128, 128, 128)])
    subvol = volume[box_to_slicing(*box)].copy()
    subvol[0:32, 0:32, 0:32] = 0
    subvol[32:64, 32:64, 32:64] = 0
    subvol[32:64, 32:64, 64:96] = 0

    #
    # Block-aligned write
    #
    service = ZarrVolumeService(config)
    service.write_subvolume(subvol, box[0])
    d = config["zarr"]["path"] + '/' + config["zarr"]["dataset"]
    _, subdirs, files = next(os.walk(d))

    empty_chunks = {(0,0,0), (1,1,1), (1,1,2)}
    nonempty_chunks = {*np.ndindex(4,4,4)} - empty_chunks
    for cz, cy, cx in nonempty_chunks:
        assert os.path.exists(f'{d}/{cz}/{cy}/{cx}')
    for cz, cy, cx in empty_chunks:
        assert not os.path.exists(f'{d}/{cz}/{cy}/{cx}')

    # Read it back.
    read_subvol = service.get_subvolume(box)
    assert (read_subvol == subvol).all()

    #
    # Again, but with a non-block aligned write
    #
    box[0] += (1, 1, 1)
    subvol = subvol[1:, 1:, 1:]

    service = ZarrVolumeService(config)
    service.write_subvolume(subvol, box[0])
    d = config["zarr"]["path"] + '/' + config["zarr"]["dataset"]
    _, subdirs, files = next(os.walk(d))

    empty_chunks = {(0,0,0), (1,1,1), (1,1,2)}
    nonempty_chunks = {*np.ndindex(4,4,4)} - empty_chunks
    for cz, cy, cx in nonempty_chunks:
        assert os.path.exists(f'{d}/{cz}/{cy}/{cx}')
    for cz, cy, cx in empty_chunks:
        assert not os.path.exists(f'{d}/{cz}/{cy}/{cx}')

    # Read it back.
    read_subvol = service.get_subvolume(box)
    assert (read_subvol == subvol).all()


if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', 'tests.volumes.test_zarr_volume_service']
    #args += ['-k', 'write_empty_blocks']
    pytest.main(args)
