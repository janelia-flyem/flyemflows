import pickle
import tempfile

import h5py
import numpy as np
import pandas as pd

import pytest

from neuclease.util import box_to_slicing

from flyemflows.volumes import Hdf5VolumeService, LabelmappedVolumeService


@pytest.fixture(scope='module')
def setup_labelmap_test():
    test_dir = tempfile.mkdtemp()
    test_file = f'{test_dir}/mapped-volume-test.h5'
    
    full_volume = np.random.randint(100, size=(256,256,256), dtype=np.uint64)
    with h5py.File(test_file, 'w') as f:
        f['volume'] = full_volume

    box_zyx = np.array([[0,0,0], [100,200,256]])
    box_xyz = box_zyx[:,::-1]

    RAW_VOLUME_DATA = full_volume[box_to_slicing(*box_zyx)]
    
    VOLUME_CONFIG = {
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
    h5_reader = Hdf5VolumeService(VOLUME_CONFIG)
    assert (h5_reader.bounding_box_zyx == box_zyx).all()
    full_from_h5 = h5_reader.get_subvolume(h5_reader.bounding_box_zyx)
    assert full_from_h5.shape == (*(box_zyx[1] - box_zyx[0]),)
    assert (full_from_h5 == RAW_VOLUME_DATA).all()

    mapping_path = f'{test_dir}/mapping.csv'
    mapping = pd.DataFrame({'orig': np.arange(100), 'body': np.arange(100) + 1000})
    mapping.to_csv(mapping_path, index=False, header=True)

    labelmap_config = {
        "file": mapping_path,
        "file-type": "label-to-body"
    }

    expected_vol = RAW_VOLUME_DATA + 1000

    return RAW_VOLUME_DATA, expected_vol, labelmap_config, full_from_h5, h5_reader, test_dir


def test_api(setup_labelmap_test):
    _raw_volume, _expected_vol, labelmap_config, _full_from_h5, h5_reader, test_dir = setup_labelmap_test
    mapped_reader = LabelmappedVolumeService(h5_reader, labelmap_config)
    assert mapped_reader.base_service == h5_reader
    assert len(mapped_reader.service_chain) == 2
    assert mapped_reader.service_chain[0] == mapped_reader
    assert mapped_reader.service_chain[1] == h5_reader


def test_read(setup_labelmap_test):
    _raw_volume, expected_vol, labelmap_config, _full_from_h5, h5_reader, test_dir = setup_labelmap_test
    mapped_reader = LabelmappedVolumeService(h5_reader, labelmap_config)

    assert (mapped_reader.bounding_box_zyx == h5_reader.bounding_box_zyx).all()
    assert (mapped_reader.preferred_message_shape == h5_reader.preferred_message_shape).all()
    assert mapped_reader.block_width == h5_reader.block_width
    assert mapped_reader.dtype == h5_reader.dtype

    full_mapped = mapped_reader.get_subvolume(mapped_reader.bounding_box_zyx)
    assert (full_mapped == expected_vol).all()
    assert full_mapped.flags.c_contiguous


def test_read_with_zero_default(setup_labelmap_test):
    raw_volume, expected_vol, _labelmap_config, _full_from_h5, h5_reader, test_dir = setup_labelmap_test

    mapping_path = f'{test_dir}/incomplete-mapping.csv'
    mapping = pd.DataFrame({'orig': np.arange(90), 'body': np.arange(90) + 1000})
    mapping.to_csv(mapping_path, index=False, header=True)
    labelmap_config = {
        "file": mapping_path,
        "file-type": "label-to-body",
        "missing-value-mode": "zero"
    }

    mapped_reader = LabelmappedVolumeService(h5_reader, labelmap_config)
    expected_vol = np.where(raw_volume < 90, expected_vol, 0)

    assert (mapped_reader.bounding_box_zyx == h5_reader.bounding_box_zyx).all()
    assert (mapped_reader.preferred_message_shape == h5_reader.preferred_message_shape).all()
    assert mapped_reader.block_width == h5_reader.block_width
    assert mapped_reader.dtype == h5_reader.dtype

    full_mapped = mapped_reader.get_subvolume(mapped_reader.bounding_box_zyx)
    assert (full_mapped == expected_vol).all()
    assert full_mapped.flags.c_contiguous


@pytest.mark.xfail
def test_write(setup_labelmap_test):
    assert False, "To-do: Implement a write test."


def test_pickle(setup_labelmap_test):
    _raw_volume, expected_vol, labelmap_config, _full_from_h5, h5_reader, test_dir = setup_labelmap_test
    mapped_reader = LabelmappedVolumeService(h5_reader, labelmap_config)
    p = pickle.dumps(mapped_reader)
    unpickled_reader = pickle.loads(p)

    full_mapped = unpickled_reader.get_subvolume(mapped_reader.bounding_box_zyx)
    assert (full_mapped == expected_vol).all()
    assert full_mapped.flags.c_contiguous


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.volumes.test_labelmapped_volume_service'])
