import tempfile
import pytest

import numpy as np

from neuclease.util import box_to_slicing
from flyemflows.volumes import TensorStoreVolumeService

try:
    import tensorstore as ts
    _have_tensorstore = True
except KeyError:
    _have_tensorstore = False


@pytest.mark.skipif(not _have_tensorstore,
                    reason="Skipping TensorStore test: tensorstore isn't installed")
def test_read(disable_auto_retry):
    """
    Minimal test to exercise the TensorStoreVolumeService.
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
                    'path': 'v1.1/segmentation',
                },
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
    service = TensorStoreVolumeService(config)

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


@pytest.mark.skipif(not _have_tensorstore,
                    reason="Skipping TensorStore test: tensorstore isn't installed")
def test_write_seg(disable_auto_retry):
    tmpdir = tempfile.mkdtemp()
    print(tmpdir)

    # Modeled on examples from this page:
    # https://google.github.io/tensorstore/driver/neuroglancer_precomputed/index.html#id12
    # plus sharding settings from the hemibrain.
    config = {
        "tensorstore": {
            "spec": {
                "driver": "neuroglancer_precomputed",
                "kvstore": {
                    "driver": "file",
                    "path": f"{tmpdir}/test-tensorstore-volume-service/test_write_seg/",
                },
                "create": True,
                "open": True,
                "multiscale_metadata": {
                    "type": "segmentation",
                    "data_type": "uint64",
                    "num_channels": 1
                },
                # Can't actually supply scale_index when creating a new scale...
                #"scale_index": 0,
                "scale_metadata": {
                    "size": [1000, 500, 300],
                    "encoding": "compressed_segmentation",
                    "compressed_segmentation_block_size": [8, 8, 8],
                    "chunk_size": [64, 64, 64],
                    "resolution": [8, 8, 8],
                    'sharding': {
                        '@type': 'neuroglancer_uint64_sharded_v1',
                        'data_encoding': 'gzip',
                        'hash': 'identity',
                        'minishard_bits': 6,
                        'minishard_index_encoding': 'gzip',
                        'preshift_bits': 9,
                        'shard_bits': 15,
                    },
                },
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
    service = TensorStoreVolumeService(config)

    def check_vol(box_zyx, scale):
        box_zyx = np.asarray(box_zyx)

        # raw volume handle
        #spec = dict(config["tensorstore"]["spec"])
        #spec['scale_index'] = scale
        #context = ts.Context(config["tensorstore"]["context"])
        #store = ts.open(spec, read=True, write=True, context=context).result()
        store = service.store(scale)
        store_box = np.array([store.spec().domain.inclusive_min[:3][::-1],
                              store.spec().domain.exclusive_max[:3][::-1]])

        # Just verify that the 'service' wrapper is consistent with the low-level handle
        assert service.dtype == store.dtype.numpy_dtype
        assert (service.bounding_box_zyx // (2**scale) == store_box).all(), \
            f"{service.bounding_box_zyx.tolist()} != {store_box.tolist()}"

        if scale == 0:
            # Service INSERTS geometry into config if necessary
            assert config["geometry"]["bounding-box"] == store_box[:,::-1].tolist()

        # FIXME: This is lazy as heck -- I should use a non-zero offset!
        data_zyx = np.random.randint(10, size=box_zyx[1] - box_zyx[0], dtype=np.uint64)
        data_czyx = data_zyx[None, ...]
        store[box_to_slicing(*box_zyx[:, ::-1])].write(data_czyx.transpose()).result()

        store_subvol = store[box_to_slicing(*box_zyx[:, ::-1])].read(order='F').result().transpose()
        assert store_subvol.any(), "Volume from raw API is all zeros; this is a bad test"

        subvol = service.get_subvolume(box_zyx, scale)
        assert subvol.any(), "Volume from service is all zeros"

        assert (subvol.shape == (box_zyx[1] - box_zyx[0])).all()
        assert (subvol == store_subvol).all()

    check_vol([(0, 0, 0), (256, 256, 256)], 0)
    check_vol([(0, 0, 0), (128, 128, 128)], 1)


def test_write_jpeg(disable_auto_retry):
    """
    Same as above, but using jpeg encoding
    """
    tmpdir = tempfile.mkdtemp()
    print(tmpdir)

    # Modeled on examples from this page:
    # https://google.github.io/tensorstore/driver/neuroglancer_precomputed/index.html#id12
    # plus sharding settings from the hemibrain grayscale.
    config = {
        "tensorstore": {
            "spec": {
                "driver": "neuroglancer_precomputed",
                "kvstore": {
                    "driver": "file",
                    "path": f"{tmpdir}/test-tensorstore-volume-service/test_write_jpeg/",
                },
                "create": True,
                "open": True,
                "multiscale_metadata": {
                    "type": "image",
                    "data_type": "uint8",
                    "num_channels": 1
                },
                # Can't actually supply scale_index when creating a new scale...
                #"scale_index": 0,
                "scale_metadata": {
                    "size": [1000, 500, 300],
                    "encoding": "jpeg",
                    "chunk_size": [64, 64, 64],
                    "resolution": [8, 8, 8],
                    'sharding': {
                        '@type': 'neuroglancer_uint64_sharded_v1',
                        'data_encoding': 'gzip',
                        'hash': 'identity',
                        'minishard_bits': 6,
                        'minishard_index_encoding': 'gzip',
                        'preshift_bits': 9,
                        'shard_bits': 15,
                    },
                },
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
    service = TensorStoreVolumeService(config)

    def check_vol(box_zyx, scale):
        box_zyx = np.asarray(box_zyx)

        # raw volume handle
        #spec = dict(config["tensorstore"]["spec"])
        #spec['scale_index'] = scale
        #context = ts.Context(config["tensorstore"]["context"])
        #store = ts.open(spec, read=True, write=True, context=context).result()
        store = service.store(scale)
        store_box = np.array([store.spec().domain.inclusive_min[:3][::-1],
                              store.spec().domain.exclusive_max[:3][::-1]])

        # Just verify that the 'service' wrapper is consistent with the low-level handle
        assert service.dtype == store.dtype.numpy_dtype
        assert (service.bounding_box_zyx // (2**scale) == store_box).all(), \
            f"{service.bounding_box_zyx.tolist()} != {store_box.tolist()}"

        if scale == 0:
            # Service INSERTS geometry into config if necessary
            assert config["geometry"]["bounding-box"] == store_box[:,::-1].tolist()

        # FIXME: This is lazy as heck -- I should use a non-zero offset!
        data_zyx = np.random.randint(10, size=box_zyx[1] - box_zyx[0], dtype=np.uint8)
        data_czyx = data_zyx[None, ...]
        store[box_to_slicing(*box_zyx[:, ::-1])].write(data_czyx.transpose()).result()

        store_subvol = store[box_to_slicing(*box_zyx[:, ::-1])].read(order='F').result().transpose()
        assert store_subvol.any(), "Volume from raw API is all zeros; this is a bad test"

        subvol = service.get_subvolume(box_zyx, scale)
        assert subvol.any(), "Volume from service is all zeros"

        assert (subvol.shape == (box_zyx[1] - box_zyx[0])).all()
        assert (subvol == store_subvol).all()

    check_vol([(0, 0, 0), (256, 256, 256)], 0)
    check_vol([(0, 0, 0), (128, 128, 128)], 1)



if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', 'tests.volumes.test_tensorstore_volume_service']
    #args += ['-k', 'write_jpeg']
    pytest.main(args)
