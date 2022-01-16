import tempfile

import h5py
import numpy as np

import pytest

from neuclease.util import box_to_slicing
from flyemflows.volumes import Hdf5VolumeService, TranslatedVolumeService
from flyemflows.volumes.adapters.scaled_volume_service import ScaledVolumeService

@pytest.fixture
def setup_translated_volume_service():
    test_dir = tempfile.mkdtemp()
    test_file = f'{test_dir}/scaled-volume-test.h5'

    full_volume = np.random.randint(255, size=(256,256,256))
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
            "available-scales": [0]  # Ensure only the first scale is used.
        }
    }

    # First, hdf5 alone
    h5_reader = Hdf5VolumeService(VOLUME_CONFIG)
    assert (h5_reader.bounding_box_zyx == box_zyx).all()
    full_from_h5 = h5_reader.get_subvolume(h5_reader.bounding_box_zyx)
    assert full_from_h5.shape == (*(box_zyx[1] - box_zyx[0]),)
    assert (full_from_h5 == RAW_VOLUME_DATA).all()

    return RAW_VOLUME_DATA, VOLUME_CONFIG, full_from_h5, h5_reader


def test_full_volume(setup_translated_volume_service):
    raw_volume, volume_config, _full_from_h5, _h5_reader = setup_translated_volume_service

    # First, h5 alone
    h5_reader = Hdf5VolumeService(volume_config)
    assert (h5_reader.bounding_box_zyx == [(0,0,0), (100,200,256)]).all()
    full_from_h5 = h5_reader.get_subvolume(h5_reader.bounding_box_zyx)
    assert full_from_h5.shape == raw_volume.shape
    assert (full_from_h5 == raw_volume).all()

    # Check API
    translated_reader = TranslatedVolumeService(h5_reader, [0, 0, 0])
    assert translated_reader.base_service == h5_reader
    assert len(translated_reader.service_chain) == 2
    assert translated_reader.service_chain[0] == translated_reader
    assert translated_reader.service_chain[1] == h5_reader

    # Now use translated reader, but with zero translation
    translated_reader = TranslatedVolumeService(h5_reader, [0, 0, 0])
    assert (translated_reader.bounding_box_zyx == h5_reader.bounding_box_zyx).all()
    assert (translated_reader.preferred_message_shape == h5_reader.preferred_message_shape).all()
    assert translated_reader.block_width == h5_reader.block_width
    assert translated_reader.dtype == h5_reader.dtype

    full_translated = translated_reader.get_subvolume(translated_reader.bounding_box_zyx)
    assert (full_translated == full_from_h5).all()
    assert full_translated.flags.c_contiguous

    # Now translate...
    tx, ty, tz = (100, 200, 300)
    translated_reader = TranslatedVolumeService(h5_reader, (tx, ty, tz))
    assert (translated_reader.bounding_box_zyx == h5_reader.bounding_box_zyx + (tz, ty, tx)).all()
    assert (translated_reader.preferred_message_shape == h5_reader.preferred_message_shape).all()
    assert translated_reader.block_width == h5_reader.block_width
    assert translated_reader.dtype == h5_reader.dtype

    full_translated = translated_reader.get_subvolume(translated_reader.bounding_box_zyx)
    assert (full_translated == full_from_h5).all()
    assert full_translated.flags.c_contiguous

    # TODO: Test write...


def test_multiscale(setup_translated_volume_service):
    _raw_volume, volume_config, _full_from_h5, _h5_reader = setup_translated_volume_service

    SCALE = 2
    h5_reader = Hdf5VolumeService(volume_config)
    h5_reader = ScaledVolumeService(h5_reader, 0)

    # No translation
    translated_reader = TranslatedVolumeService(h5_reader, (0, 0, 0))
    from_h5 = h5_reader.get_subvolume(h5_reader.bounding_box_zyx // 2**SCALE, SCALE)
    from_transposed = translated_reader.get_subvolume(h5_reader.bounding_box_zyx // 2**SCALE, SCALE)
    assert (from_transposed == from_h5).all()

    # Now translate...
    tx, ty, tz = (100, 200, 300)
    translated_reader = TranslatedVolumeService(h5_reader, (tx, ty, tz))
    from_h5 = h5_reader.get_subvolume(h5_reader.bounding_box_zyx // 2**SCALE, SCALE)
    from_translated = translated_reader.get_subvolume(translated_reader.bounding_box_zyx // 2**SCALE, SCALE)
    assert (from_translated == from_h5).all()


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.volumes.test_translated_volume_service'])
