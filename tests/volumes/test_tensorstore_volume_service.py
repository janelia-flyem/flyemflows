import pytest

import numpy as np

from neuclease.util import box_to_slicing
from flyemflows.volumes import TensorStoreVolumeServiceReader

try:
    import tensorstore as ts
    _have_tensorstore = True
except KeyError:
    _have_tensorstore = False


@pytest.mark.skipif(not _have_tensorstore,
                    reason="Skipping TensorStore test: tensorstore isn't installed")
def test_read(disable_auto_retry):
    """
    Minimal test to exercise the TensorStoreVolumeServiceReader.
    Reads a tiny subvolume from a known neuroglancer_precomputed volume instance.

    If this is ever deleted or becomes unavailable,
    we'll have to find a different volume to test with.
    """
    config = {
        "tensorstore": {
            "spec": {
                'driver': 'neuroglancer_precomputed',
                'kvstore': {
                    'driver': 'gcs',
                    'bucket': 'neuroglancer-janelia-flyem-hemibrain',
                },
                'path': 'v1.1/segmentation',
                'open': True,
                'data_copy_concurrency': {"limit": 1},  # or specify in context
                'recheck_cached_metadata': False,
                'recheck_cached_data': False,
            },
            "context": {
                "cache_pool": {"total_bytes_limit": 8*(512**3)},
                "data_copy_concurrency": {"limit": 8},
                'file_io_concurrency': {'limit': 1}
            },
        },
        "geometry": {}
    }
    service = TensorStoreVolumeServiceReader(config)

    def check_vol(box_zyx, scale):
        # raw volume handle
        spec = dict(config["tensorstore"]["spec"])
        spec['scale_index'] = scale
        context = ts.Context(config["tensorstore"]["context"])
        store = ts.open(spec, read=True, write=False, context=context).result()
        store_box = np.array([store.spec().domain.inclusive_min[:3][::-1],
                            store.spec().domain.exclusive_max[:3][::-1]])

        # Just verify that the 'service' wrapper is consistent with the low-level handle
        assert service.dtype == store.dtype.numpy_dtype
        assert (service.bounding_box_zyx // (2**scale) == store_box).all(), \
            f"{service.bounding_box_zyx.tolist()} != {store_box.tolist()}"

        if scale == 0:
            # Service INSERTS geometry into config if necessary
            assert config["geometry"]["bounding-box"] == store_box[:,::-1].tolist()

        store_subvol = store[box_to_slicing(*box_zyx[:, ::-1])].read(order='F').result().transpose()
        assert store_subvol.any(), "Volume from raw API is all zeros; this is a bad test"

        subvol = service.get_subvolume(box_zyx, scale)
        assert subvol.any(), "Volume from service is all zeros"

        assert (subvol.shape == (box_zyx[1] - box_zyx[0])).all()
        assert (subvol == store_subvol).all()

    # Use a non-aligned box; should still work.
    start_xyz = np.array([20480+1, 20480+2, 20480+3])
    start_zyx = start_xyz[::-1]
    box_zyx = np.array([start_zyx, start_zyx + (100, 200, 300)])

    # Check a few scales
    check_vol(box_zyx, 0)
    check_vol(box_zyx // (2**1), 1)
    check_vol(box_zyx // (2**2), 2)


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.volumes.test_tensorstore_volume_service'])
