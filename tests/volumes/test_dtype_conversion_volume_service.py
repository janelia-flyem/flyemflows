import tempfile

import h5py
import numpy as np
import pytest

from flyemflows.volumes import (
    VolumeService,
    Hdf5VolumeService,
    DtypeConversionVolumeService,
)


@pytest.fixture
def setup_uint64_volume():
    test_dir = tempfile.mkdtemp()
    test_file = f"{test_dir}/dtype-conv-test.h5"

    # Values that all fit in uint32 so the default 'error' overflow-mode is happy.
    full_volume = np.random.randint(0, 2**16, size=(32, 32, 32), dtype=np.uint64)
    with h5py.File(test_file, "w") as f:
        f["volume"] = full_volume

    volume_config = {
        "hdf5": {
            "path": test_file,
            "dataset": "volume",
        },
        "geometry": {
            "bounding-box": [[0, 0, 0], [32, 32, 32]],
            "available-scales": [0],
        },
    }
    return full_volume, volume_config


def _make_dtype_adapter(reader, target, apply_when="reading-and-writing", overflow_mode="error"):
    return DtypeConversionVolumeService(
        reader,
        {"dtype": target, "apply-when": apply_when, "overflow-mode": overflow_mode},
    )


def test_read_narrowing(setup_uint64_volume):
    """Reads cast uint64 -> uint32; geometry/scales pass through unchanged."""
    raw, cfg = setup_uint64_volume
    reader = Hdf5VolumeService(cfg)
    adapter = _make_dtype_adapter(reader, "uint32")

    assert adapter.dtype == np.dtype("uint32")
    assert adapter.base_service is reader
    assert (adapter.bounding_box_zyx == reader.bounding_box_zyx).all()
    assert (adapter.preferred_message_shape == reader.preferred_message_shape).all()
    assert adapter.block_width == reader.block_width
    assert adapter.available_scales == reader.available_scales

    out = adapter.get_subvolume(adapter.bounding_box_zyx)
    assert out.dtype == np.dtype("uint32")
    assert (out == raw.astype(np.uint32)).all()


def test_read_widening_roundtrip(setup_uint64_volume):
    """Widening (uint64 -> uint64 of a uint32-truncated value) round-trips."""
    raw, cfg = setup_uint64_volume
    reader = Hdf5VolumeService(cfg)
    narrow = _make_dtype_adapter(reader, "uint32")
    out_narrow = narrow.get_subvolume(narrow.bounding_box_zyx)

    widen = _make_dtype_adapter(narrow, "uint64")
    assert widen.dtype == np.dtype("uint64")
    out_wide = widen.get_subvolume(widen.bounding_box_zyx)
    assert out_wide.dtype == np.dtype("uint64")
    assert (out_wide == out_narrow.astype(np.uint64)).all()
    assert (out_wide == raw).all()


def test_read_overflow_raises():
    """In 'error' mode, a value too large for the target dtype must raise."""
    test_dir = tempfile.mkdtemp()
    test_file = f"{test_dir}/dtype-conv-overflow.h5"
    vol = np.zeros((4, 4, 4), dtype=np.uint64)
    vol[0, 0, 0] = (1 << 33)  # too big for uint32
    with h5py.File(test_file, "w") as f:
        f["volume"] = vol

    cfg = {
        "hdf5": {"path": test_file, "dataset": "volume"},
        "geometry": {"bounding-box": [[0, 0, 0], [4, 4, 4]], "available-scales": [0]},
    }
    adapter = _make_dtype_adapter(Hdf5VolumeService(cfg), "uint32", overflow_mode="error")
    with pytest.raises(RuntimeError, match="out of range"):
        adapter.get_subvolume(adapter.bounding_box_zyx)


def test_read_overflow_permitted():
    """In 'permit' mode, narrowing silently truncates without raising."""
    test_dir = tempfile.mkdtemp()
    test_file = f"{test_dir}/dtype-conv-permit.h5"
    vol = np.zeros((4, 4, 4), dtype=np.uint64)
    vol[0, 0, 0] = (1 << 33) | 7  # low 32 bits == 7
    with h5py.File(test_file, "w") as f:
        f["volume"] = vol

    cfg = {
        "hdf5": {"path": test_file, "dataset": "volume"},
        "geometry": {"bounding-box": [[0, 0, 0], [4, 4, 4]], "available-scales": [0]},
    }
    adapter = _make_dtype_adapter(Hdf5VolumeService(cfg), "uint32", overflow_mode="permit")
    out = adapter.get_subvolume(adapter.bounding_box_zyx)
    assert out.dtype == np.dtype("uint32")
    assert out[0, 0, 0] == 7


def test_write_narrowing(setup_uint64_volume):
    """Writes cast user-supplied uint32 down to the underlying uint64 store."""
    raw, cfg = setup_uint64_volume
    # Use a separate file so we don't clobber the read fixture.
    test_dir = tempfile.mkdtemp()
    cfg = {**cfg, "hdf5": {**cfg["hdf5"], "path": f"{test_dir}/write_target.h5"}}
    cfg["hdf5"]["writable"] = True

    # Create the underlying h5 dataset with zeros so the writer can target it.
    with h5py.File(cfg["hdf5"]["path"], "w") as f:
        f.create_dataset("volume", shape=(32, 32, 32), dtype=np.uint64, chunks=(16, 16, 16))

    writer = Hdf5VolumeService(cfg)
    adapter = _make_dtype_adapter(writer, "uint32")

    user_data = (raw.astype(np.uint32))  # uint32 view of the original uint64 data
    adapter.write_subvolume(user_data, (0, 0, 0))

    # Read back from underlying writer (sees uint64).
    stored = writer.get_subvolume(writer.bounding_box_zyx)
    assert stored.dtype == np.dtype("uint64")
    assert (stored == raw).all()

    # And the adapter sees uint32 still.
    read_back = adapter.get_subvolume(adapter.bounding_box_zyx)
    assert read_back.dtype == np.dtype("uint32")
    assert (read_back == user_data).all()


def test_apply_when_reading_only(setup_uint64_volume):
    """apply-when=reading: writes pass through without casting."""
    raw, cfg = setup_uint64_volume
    reader = Hdf5VolumeService(cfg)
    adapter = _make_dtype_adapter(reader, "uint32", apply_when="reading")

    assert adapter.apply_when_reading
    assert not adapter.apply_when_writing

    out = adapter.get_subvolume(adapter.bounding_box_zyx)
    assert out.dtype == np.dtype("uint32")
    assert (out == raw.astype(np.uint32)).all()


def test_create_from_config(setup_uint64_volume):
    """The adapter is wired up by VolumeService.create_from_config when configured."""
    raw, cfg = setup_uint64_volume
    full_cfg = {**cfg, "adapters": {"convert-dtype": {"dtype": "uint32"}}}
    service = VolumeService.create_from_config(full_cfg)

    assert isinstance(service, DtypeConversionVolumeService)
    assert service.dtype == np.dtype("uint32")
    out = service.get_subvolume(service.bounding_box_zyx)
    assert out.dtype == np.dtype("uint32")
    assert (out == raw.astype(np.uint32)).all()


def test_create_from_config_noop_when_empty(setup_uint64_volume):
    """An empty 'dtype' means the adapter is not added at all."""
    _, cfg = setup_uint64_volume
    full_cfg = {**cfg, "adapters": {"convert-dtype": {"dtype": ""}}}
    service = VolumeService.create_from_config(full_cfg)
    assert not isinstance(service, DtypeConversionVolumeService)
    assert service.dtype == np.dtype("uint64")


if __name__ == "__main__":
    pytest.main(["-s", "--tb=native", "--pyargs", "tests.volumes.test_dtype_conversion_volume_service"])
