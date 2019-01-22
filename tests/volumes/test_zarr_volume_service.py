import os
import pytest

import zarr
import numpy as np

from neuclease.util import box_to_slicing

from flyemflows.volumes import ZarrVolumeService

@pytest.fixture(scope="module")
def volume_setup():
    path = "/tmp/test_zarr_service_testvol.zarr"
    dataset = "/some/volume"
    
    config = {
        "zarr": {
            "path": path,
            "dataset": dataset
        },
        "geometry": {}
    }
    
    volume = np.random.randint(100, size=(512, 256, 128))
    f = zarr.open(path, 'w')
    f.create_dataset(dataset, data=volume, chunks=(64,64,64))

    return config, volume


def test_read(volume_setup):
    config, volume = volume_setup
    
    service = ZarrVolumeService(config)    
    assert (service.bounding_box_zyx == [(0,0,0),volume.shape]).all()
    assert service.dtype == volume.dtype

    # Service INSERTS geometry into config if necessary
    assert config["geometry"]["bounding-box"] == [[0,0,0], list(volume.shape[::-1])]
    assert config["zarr"]["dtype"] == volume.dtype.name
    
    box = [(30,40,50), (50,60,70)]
    subvol = service.get_subvolume(box)
    
    assert (subvol == volume[box_to_slicing(*box)]).all()


def test_write(volume_setup):
    config, volume = volume_setup
    config["zarr"]["path"] = "/tmp/test_zarr_service_testvol_WRITE.zarr"
    if os.path.exists(config["zarr"]["path"]):
        os.unlink(config["zarr"]["path"])

    # Can't initialize service if file doesn't exist
    with pytest.raises(RuntimeError) as excinfo:
        ZarrVolumeService(config)
    assert 'writable' in str(excinfo.value)

    # After setting writable=true, we can initialize the service.    
    assert not os.path.exists(config["zarr"]["path"])
    config["zarr"]["writable"] = True

    # Write some data
    box = [(30,40,50), (50,60,70)]
    subvol = volume[box_to_slicing(*box)]
    service = ZarrVolumeService(config)
    service.write_subvolume(subvol, box[0])

    # Read it back.
    subvol = service.get_subvolume(box)
    assert (subvol == volume[box_to_slicing(*box)]).all()


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.volumes.test_zarr_volume_service'])
