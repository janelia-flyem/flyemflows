import os
import copy
import pickle
import logging
import threading
import multiprocessing as mp

import numpy as np

from confiddler import validate, emit_defaults
from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, box_to_slicing, box_intersection

from ..util import auto_retry, replace_default_entries
from . import VolumeServiceWriter, GeometrySchema, SegmentationAdapters

logger = logging.getLogger(__name__)

TensorStoreSpecSchema = \
{
    "description": "TensorStore Spec",
    "type": "object",
    "default": {},
    "required": ["driver", "kvstore"],
    # "additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "driver": {
            "description": "type of volume",
            "type": "string",
            # For now, we support only ng precomputed, not zarr or n5.
            # We would need to be more careful about fortran/C ordering to handle other sources.
            "enum": ["neuroglancer_precomputed"],
            "default": "neuroglancer_precomputed"
        },
        "kvstore": {
            "type": "object",
            "default": {},
            "required": ["driver"],
            "properties": {
                "driver": {
                    "type": "string",
                    "enum": ["gcs", "file"],
                    #"default": "file"  # Must not have a default value, to ensure that the default config doesn't meet criteria for a valid config.
                },
                "bucket": {
                    "description": "bucket name",
                    "type": "string",
                    "default": ""
                },
                "path": {
                    "description": "Path on the filesystem or within the bucket, e.g. /tmp/foo/",
                    "type": "string",
                    "default": ""
                },
            },
        },
        "path": {
            "description": "DEPRECATED.  DO NOT USE.  (Use kvstore.path instead.)",
            "type": "string",
            "default": "",
        },
        "recheck_cached_metadata": {
            "description": "When to check that the metadata hasn't changed.",
            "oneOf": [
                {"type": "string", "enum": ["open"]},  # When the store is opened
                {"type": "boolean"},
                {"type": "integer"}
            ],
            "default": False
        },
        "recheck_cached_data": {
            "description": "When to check that cached data hasn't changed.",
            "oneOf": [
                {"type": "string", "enum": ["open"]},  # When the store is opened
                {"type": "boolean"},
                {"type": "integer"}
            ],
            "default": False
        }

        # Note:
        #   scale_index is configured on the fly, below.
        #   don't add it here in the config.
        # "scale_index": {
        #    "type": "integer"
        # }
    }
}

TensorStoreContextSchema = {
    "description": "TensorStore Spec",
    "type": "object",
    "required": [],
    "default": {},
    "properties": {
        "cache_pool": {
            "type": "object",
            "default": {},
            "properties": {
                "total_bytes_limit": {
                    "type": "integer",
                    "default": 0
                }
            }
        },
        "data_copy_concurrency": {
            "type": "object",
            "default": {},
            "properties": {
                "limit": {
                    "default": 1,  # TensorStore's default is 'shared' but in a
                                   # cluster job, presumably we want one per worker.
                    "oneOf": [
                        {"type": "integer"},
                        {"type": "string", "enum": ["shared"]}
                    ]
                }
            }
        },
        "file_io_concurrency": {
            "type": "object",
            "default": {},
            "properties": {
                "limit": {
                    "default": 1,  # TensorStore's default is 'shared' but in a
                                   # cluster job, presumably we want one per worker.
                    "oneOf": [
                        {"type": "integer"},
                        {"type": "string", "enum": ["shared"]}
                    ]
                }
            }
        },
    }
}

TensorStoreServiceSchema = {
    "description": "TensorStore Service settings. Only some of the valid properties are listed here.  For the full schema, see the TensorStore docs.",
    "type": "object",
    "default": {},
    "required": ["spec", "context"],
    # "additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "spec": TensorStoreSpecSchema,
        "context": TensorStoreContextSchema,
        "out-of-bounds-access": {
            "description": "If 'forbid', any out-of-bounds read/write is an error.\n"
                           "If 'permit', out-of-bounds reads are permitted, and filled with zeros, and out-of-bounds writes are merely ignored.\n"
                           "If 'permit-empty', out-of-bounds writes are permitted (and ignored) only if the out-of-bounds voxels are zero.\n"
                           "Otherwise, they're an error.",
            "type": "string",
            "enum": ["forbid", "permit", "permit-empty"],
            "default": "permit-empty"
        },
        "reinitialize-via": {
            "type": "string",
            "enum": ["unpickle", "reopen"],
            "default": "reopen"  # 'unpickle' doesn't seem to work consistently, at least in write mode.
        },
        "read-as-dtype": {
            "description":
                "A flyemflows-specific setting. Not used when writing subvolumes.\n"
                "Converts fetched subvolumes to this dtype immediately after they are fetched from the cloud,\n"
                "regardless of the actual dtype in storage.\n",
            "default": None,
            "oneOf": [
                {
                    "type": "null"
                },
                {
                    "type": "string",
                    "enum": ['bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64'],
                    "default": "uint64"
                },
            ]
        },
        "subprocess-timeout": {
            "description": "If nonzero, make requests in a subprocess and retry them if they take longer than this many seconds.",
            "type": "number",
            "default": 0.0
        }
    }
}

TensorStoreVolumeSchema = \
{
    "description": "Describes a volume accessible via TensorStore.",
    "type": "object",
    "default": {},
    # "additionalProperties": False,
    "properties": {
        "tensorstore": TensorStoreServiceSchema,
        "geometry": GeometrySchema,
        "adapters": SegmentationAdapters
    }
}


class TensorStoreVolumeService(VolumeServiceWriter):
    """
    A wrapper around the TensorStore API to implement the VolumeServiceWriter API.

    Warning:
        Currently, this class requires you to supply most of the tensorstore dataset configuration explicitly,
        even if you've also supplied redundant config paramters in the 'geometry' section of the volume config.


    Note:
        To double-check the shard size for a given chunk_size, preshift bits, etc., try:
        cloudvolume.datasource.precomputed.ShardingSpecification(...).image_shard_shape()

    Here's an example config:



        tensorstore:
            spec: {
                "driver": "neuroglancer_precomputed",
                "kvstore": {
                    "driver": "file",
                    "path": "/tmp/foobar-volume/",
                },
                "create": true,
                "open": true,
                "multiscale_metadata": {
                    "type": "segmentation",
                    "data_type": "uint64",
                    "num_channels": 1
                },
                "scale_metadata": {
                    "size": [1000, 500, 300],
                    "encoding": "compressed_segmentation",
                    "compressed_segmentation_block_size": [8, 8, 8],
                    "chunk_size": [64, 64, 64],
                    "resolution": [8, 8, 8],
                    "sharding": {
                        "@type": "neuroglancer_uint64_sharded_v1",
                        "data_encoding": "gzip",
                        "hash": "identity",
                        "minishard_bits": 6,
                        "minishard_index_encoding": "gzip",
                        "preshift_bits": 9,
                        "shard_bits": 15,
                    },
                },
                "data_copy_concurrency": {"limit": 1},
                "recheck_cached_metadata": False,
                "recheck_cached_data": False,
            }
            "context": {
                "cache_pool": {"total_bytes_limit": 8*(512**3)},
                "data_copy_concurrency": {"limit": 8},
                'file_io_concurrency': {'limit': 1}
            }
        geometry:
            bounding-box: [[0,0,0], [1000, 500, 300]]
            message-block-shape: [2048, 2048, 2048]
            block-width: 64
            available-scales: [0,1,2,3,4,5,6,7]
    """

    @classmethod
    def default_config(cls, ng_src_url=None):
        """
        Return the default config for a TensorStoreVolumeService.
        If you provide a source url, the bucket and path components will be configured.
        Otherwise, you must overwrite those entries yourself before using the config.

        Example url:

            precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation
        """
        cfg = emit_defaults(TensorStoreVolumeSchema)
        if ng_src_url:
            # Example:
            # precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation
            # But the 'precomputed://' and 'gs://' prefixes are optional.
            parts = ng_src_url.split('://')
            assert len(parts) <= 3
            if len(parts) == 3:
                assert parts[0] == 'precomputed'
            if len(parts) >= 2:
                assert parts[-2] in ('gs', 'file')

            if parts[-2] == 'gs':
                bucket, *path_parts = parts[-1].split('/')
                cfg['tensorstore']['spec']['kvstore']['driver'] = 'gcs'
                cfg['tensorstore']['spec']['kvstore']['bucket'] = bucket
                cfg['tensorstore']['spec']['kvstore']['path'] = '/'.join(path_parts)
            elif parts[-2] == 'file':
                del cfg['tensorstore']['spec']['kvstore']['bucket']
                cfg['tensorstore']['spec']['kvstore']['driver'] = 'file'
                cfg['tensorstore']['spec']['kvstore']['path'] = parts[-1]

        return cfg

    @classmethod
    def from_url(cls, ng_src_url):
        """
        Construct a TensorStoreVolumeService from a neuroglancer source url,
        such as:

            precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation

        Other than the bucket and path, the config will use default settings.
        """
        cfg = cls.default_config(ng_src_url)
        return cls(cfg)

    def __init__(self, volume_config, resource_manager_client=None):
        validate(volume_config, TensorStoreVolumeSchema, inject_defaults=True)

        if resource_manager_client is None:
            # Dummy client
            resource_manager_client = ResourceManagerClient("", 0)

        self.volume_config = volume_config

        if volume_config['tensorstore']['spec']['path']:
            raise RuntimeError("The tensorstore.spec.path property is deprecated.  Please use tensorstore.spec.kvstore.path instead.")

        try:
            # Strip 'gs://' if the user provided it.
            bucket = volume_config['tensorstore']['spec']['kvstore']['bucket']
            if bucket.startswith('precomputed://'):
                bucket = bucket[len('precomputed://'):]
            if bucket.startswith('gs://'):
                bucket = bucket[len('gs://'):]
                volume_config['tensorstore']['spec']['kvstore']['bucket'] = bucket
        except KeyError:
            pass

        if volume_config['tensorstore']['spec']['kvstore']['driver'] == 'file':
            assert volume_config['tensorstore']['spec']['kvstore']['bucket'] == ''
            # We must not include 'bucket' if we're writing to a local file
            del volume_config['tensorstore']['spec']['kvstore']['bucket']

        self._stores = {}
        store = self.store(0)
        spec = store.spec()

        block_width = volume_config["geometry"]["block-width"]
        if block_width == -1:
            block_width = spec.to_json()['scale_metadata']['chunk_size'][0]

        preferred_message_shape_zyx = np.array( volume_config["geometry"]["message-block-shape"][::-1] )
        replace_default_entries(preferred_message_shape_zyx, [256, 256, 256])

        preferred_grid_offset_zyx = np.array( volume_config["geometry"]["message-grid-offset"][::-1] )

        # Convert box from xyzc -> zyx
        store_box_zyx = np.array([spec.domain.inclusive_min, spec.domain.exclusive_max])[:, :3][:, ::-1]
        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        replace_default_entries(bounding_box_zyx, store_box_zyx)

        assert (bounding_box_zyx[0] >= store_box_zyx[0]).all() and (bounding_box_zyx[1] <= store_box_zyx[1]).all(), \
            f"Specified bounding box ({bounding_box_zyx[:, ::-1].tolist()}) extends outside the "\
            f"TensorStore volume geometry ({store_box_zyx[:, ::-1].tolist()})"

        # FIXME: Figure out how to configure this automatically.
        available_scales = list(volume_config["geometry"]["available-scales"])

        # Store members
        self._block_width = block_width
        self._bounding_box_zyx = bounding_box_zyx
        self._resource_manager_client = resource_manager_client
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._preferred_grid_offset_zyx = preferred_grid_offset_zyx
        self._available_scales = available_scales
        self._reinitialize_via = volume_config["tensorstore"]["reinitialize-via"]
        self._out_of_bounds_access = volume_config["tensorstore"]["out-of-bounds-access"]
        self._read_as_dtype = volume_config["tensorstore"]["read-as-dtype"]
        if self._read_as_dtype is None:
            self._dtype = spec.dtype.numpy_dtype
        else:
            self._dtype = np.dtype(self._read_as_dtype)

        # Overwrite config entries that we might have modified
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()
        volume_config["geometry"]["message-grid-offset"] = self._preferred_grid_offset_zyx[::-1].tolist()

        self._ensure_scales_exist()

        self._pools = {}
        self._pools_lock = threading.Lock()
        self._subprocess_timeout = volume_config["tensorstore"]["subprocess-timeout"]

    def _pool(self):
        """
        We maintain a dict of multiprocessing Pools
        (one per thread of the main process).

        This function returns the Pool we created for the current thread.
        If it doesn't exist yet (or has been discarded via _reset_pool()),
        we create a new Pool with a single Worker.
        """
        pid = os.getpid()
        thread_id = threading.current_thread().ident
        with self._pools_lock:
            if pool := self._pools.get((pid, thread_id)):
                return pool
            pool = mp.get_context('spawn').Pool(1)
            self._pools[(pid, thread_id)] = pool
            return pool

    def _reset_pool(self):
        """
        Discard the pool for the current process/thread.
        Note:
            This will simply leak the pool without closing it, but
            that should be rare enough in practice that we don't care.
        """
        pid = os.getpid()
        thread_id = threading.current_thread().ident
        with self._pools_lock:
            self._pools[(pid, thread_id)] = None
            del self._pools[(pid, thread_id)]

    def store(self, scale):
        try:
            return self._stores[scale]
        except KeyError:
            import tensorstore as ts

            # We assume the user supplied the metadata for scale 0,
            # and here we modify it for higher scales.
            spec = copy.deepcopy(self.volume_config['tensorstore']['spec'])
            context = copy.deepcopy(self.volume_config['tensorstore']['context'])

            create = spec.get('create', False)
            if create:
                # We can't actually supply scale_index when creating a new scale,
                # despite what the docs claim.
                if 'scale_index' in spec:
                    del spec['scale_index']

                shape = np.asarray(spec['scale_metadata']['size'])
                spec['scale_metadata']['size'] = (shape // 2**scale).tolist()

                res = np.asarray(spec['scale_metadata']['resolution'])
                spec['scale_metadata']['resolution'] = (res * (2**scale)).tolist()
            else:
                spec['scale_index'] = scale

            store = ts.open(spec, read=True, write=create, context=ts.Context(context)).result()
            self._stores[scale] = store
            return store

    def _ensure_scales_exist(self):
        for scale in self.available_scales:
            self.store(scale)

    def __getstate__(self):
        """
        Pickle representation.
        """
        d = self.__dict__.copy()

        # Apparently unpickling can fail intermittently due to network issues, so we
        # pickle/unpickle it explicitly so we can control the process and retry if necessary.
        d['_stores'] = pickle.dumps(d['_stores'])

        # Don't pickle the process pool or the associated lock
        d['_pools'] = {}
        d['_pools_lock'] = None

        return d

    @auto_retry(3, pause_between_tries=5.0, logging_name=__name__)
    def __setstate__(self, d):
        # We unpickle the TensorStore(s) explicitly here, to ensure
        # that errors are caught by the auto_retry decorator.
        #
        # Here's an example error we want to catch:
        #
        #   ValueError: Error opening "neuroglancer_precomputed" driver:
        #     Error reading "gs://vnc-v3-seg/rc4_wsexp/info":
        #     CURL error[16] Error in the HTTP2 framing layer
        #
        if d['_reinitialize_via'] == "reopen":
            d['_stores'] = {}
        else:
            d['_stores'] = pickle.loads(d['_stores'])

        # Initialize the pool lock.
        d['_pools_lock'] = threading.Lock()

        self.__dict__ = d

    @property
    def dtype(self):
        return self._dtype

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape_zyx

    @property
    def preferred_grid_offset(self):
        return self._preferred_grid_offset_zyx

    @property
    def block_width(self):
        return self._block_width

    @property
    def bounding_box_zyx(self):
        return self._bounding_box_zyx

    @property
    def available_scales(self):
        return self._available_scales

    @property
    def resource_manager_client(self):
        return self._resource_manager_client

    # Two levels of auto-retry:
    # If a failure is NOT 503 (e.g. a non-503 TimeoutError), restart it up to three times with a short delay.
    # If the request fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
    @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                predicate=lambda ex: '503' in str(ex) or '504' in str(ex))
    @auto_retry(3, pause_between_tries=30.0, logging_name=__name__,
                predicate=lambda ex: '503' not in str(ex) and '504' not in str(ex))
    def get_subvolume(self, box_zyx, scale=0):
        if not self._subprocess_timeout:
            return self._get_boundschecked_subvolume(box_zyx, scale)

        try:
            return (
                self._pool()
                .apply_async(self._get_boundschecked_subvolume, (box_zyx, scale))
                .get(self._subprocess_timeout)
            )
        except mp.TimeoutError:
            logger.warning(
                "TensorStoreVolumeService.get_subvolume(): TimeoutError: "
                f"{box_zyx[:, ::-1].tolist()} (XYZ), scale={scale}"
            )
            # That process could be corrupt now...
            self._reset_pool()
            raise

    def _get_boundschecked_subvolume(self, box_zyx, scale=0):
        logger.info(f"Tensorstore: Fetching {box_zyx[:, ::-1].tolist()} (XYZ)")
        box_zyx = np.asarray(box_zyx)
        full_shape_xyzc = self.store(scale).shape
        full_shape_zyx = full_shape_xyzc[-2::-1]
        clipped_box = box_intersection(box_zyx, [(0,0,0), full_shape_zyx])
        if (clipped_box == box_zyx).all():
            return self._get_subvolume_nocheck(box_zyx, scale)

        # Note that this message shows the true tenstorestore storage bounds,
        # and doesn't show the logical bounds according to global_offset (if any).
        msg = (f"TensorStore Request is out-of-bounds (XYZ): {box_zyx[:, ::-1].tolist()}"
                " relative to volume extents (XYZC): {full_shape_xyzc.tolist()}")
        if self._out_of_bounds_access in ("permit", "permit-empty"):
            logger.warning(msg)
        else:
            msg += "\nAdd 'out-of-bounds-access' to your config to allow such requests"
            raise RuntimeError(msg)

        if (clipped_box[1] - clipped_box[0] <= 0).any():
            # request is completely out-of-bounds; just return zeros
            return np.zeros(box_zyx[1] - box_zyx[0], self.dtype)

        # Request is partially out-of-bounds; read what we can, zero-fill for the rest.
        clipped_vol = self._get_subvolume_nocheck(clipped_box, scale)
        result = np.zeros(box_zyx[1] - box_zyx[0], self.dtype)
        localbox = clipped_box - box_zyx[0]
        result[box_to_slicing(*localbox)] = clipped_vol
        return result

    def _get_subvolume_nocheck(self, box_zyx, scale):
        box_zyx = np.asarray(box_zyx)
        req_bytes = 8 * np.prod(box_zyx[1] - box_zyx[0])
        try:
            resource_name = self.volume_config['tensorstore']['spec']['kvstore']['bucket']
        except KeyError:
            resource_name = self.volume_config['tensorstore']['spec']['kvstore']['path']

        with self._resource_manager_client.access_context(resource_name, True, 1, req_bytes):
            store = self.store(scale)

            # Tensorstore uses X,Y,Z conventions, so it's best to
            # request a Fortran array and transpose it ourselves.
            box_xyz = box_zyx[:, ::-1]
            vol_xyzc = store[box_to_slicing(*box_xyz)].read(order='F').result()
            vol_xyz = vol_xyzc[..., 0]
            vol_zyx = vol_xyz.transpose()

            assert (vol_zyx.shape == (box_zyx[1] - box_zyx[0])).all(), \
                f"Fetched volume_zyx shape ({vol_zyx.shape} doesn't match box_zyx {box_zyx.tolist()}"
            return vol_zyx.astype(self.dtype, copy=False)

    # Two levels of auto-retry:
    # If a failure is NOT 503 (e.g. a non-503 TimeoutError), restart it up to three times with a short delay.
    # If the request fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
    @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                predicate=lambda ex: '503' in str(ex) or '504' in str(ex))
    @auto_retry(3, pause_between_tries=30.0, logging_name=__name__,
                predicate=lambda ex: '503' not in str(ex) and '504' not in str(ex))
    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        """
        Note:
            It is the user's responsibility to ensure that the data is written in
            chunks that are aligned to the volume's native shard size (chunk_layout.write_chunk.shape).

            In [1]: import tensorstore as ts
               ...:
               ...: hemibrain_uri = 'gs://neuroglancer-janelia-flyem-hemibrain/v1.1/segmentation/'
               ...: dset = ts.open({
               ...:     'driver': 'neuroglancer_precomputed',
               ...:     'kvstore': hemibrain_uri}
               ...: ).result()
               ...:
               ...: dset.chunk_layout.write_chunk.shape
            Out[1]: (2048, 2048, 2048, 1)
        """
        if not self._subprocess_timeout:
            return self._write_boundschecked_subvolume(subvolume, offset_zyx, scale)

        try:
            return (
                self._pool()
                .apply_async(self._write_boundschecked_subvolume, (subvolume, offset_zyx, scale))
                .get(self._subprocess_timeout)
            )
        except mp.TimeoutError:
            box_zyx = np.array([offset_zyx, subvolume.shape])
            logger.warning(
                "TensorStoreVolumeService.write_subvolume(): TimeoutError: "
                f"{box_zyx[:, ::-1].tolist()} (XYZ), scale={scale}"
            )
            # That process could be corrupt now...
            self._reset_pool()
            raise

    def _write_boundschecked_subvolume(self, subvolume, offset_zyx, scale=0):
        offset_zyx = np.array(offset_zyx)
        if offset_zyx.shape != (3,):
            raise ValueError(f"offset_zyx should be a single (Z,Y,X) tuple, not {offset_zyx}")
        box_zyx = np.array([offset_zyx, offset_zyx+subvolume.shape])

        full_shape_xyzc = self.store(scale).shape
        full_shape_zyx = full_shape_xyzc[-2::-1]
        clipped_box = box_intersection(box_zyx, [(0,0,0), full_shape_zyx])
        if (clipped_box == box_zyx).all():
            # Box is fully contained within the output volume bounding box.
            self._write_subvolume_nocheck(subvolume, box_zyx[0], scale)
            return

        msg = (f"Box (XYZ): {box_zyx[:, ::-1].tolist()}"
               f" exceeds full scale-{scale} extents (XYZC) [(0,0,0), {full_shape_xyzc}]")

        if self._out_of_bounds_access == 'forbid':
            msg = "Cannot write subvolume. " + msg
            msg += "\nAdd permit-out-of-bounds to your config to allow such writes,"
            msg += " assuming the out-of-bounds portion is completely empty."
            raise RuntimeError(msg)

        subvol_copy = subvolume.copy()
        if (clipped_box[0] < clipped_box[1]).all():
            subvol_copy[box_to_slicing(*(clipped_box - box_zyx[0]))] = 0
        if self._out_of_bounds_access == 'permit-empty' and subvol_copy.any():
            msg = ("Cannot write subvolume. Box extends beyond volume storage bounds (XYZ): "
                f"{box_zyx[:, ::-1].tolist()} exceeds [(0,0,0), {full_shape_zyx[::-1]}\n"
                "and the out-of-bounds portion is not empty (contains non-zero values).\n")
            raise RuntimeError(msg)

        if (clipped_box[0] >= clipped_box[1]).any():
            # The box_intersection was invalid;
            # so the subvolume is completely outside the output volume box.
            return

        logger.warning(msg)
        clipped_subvolume = subvolume[box_to_slicing(*clipped_box - box_zyx[0])]
        self._write_subvolume_nocheck(clipped_subvolume, clipped_box[0], scale)

    def _write_subvolume_nocheck(self, subvolume, offset_zyx, scale):
        offset_czyx = np.array((0, *offset_zyx))
        shape_czyx = np.array((1, *subvolume.shape))
        box_czyx = np.array([offset_czyx, offset_czyx + shape_czyx])

        # With neuroglancer_precomputed, tensorstore uses X,Y,Z conventions,
        # regardless of the actual memory order.  So it's best to send a Fortran array.
        vol_xyzc = subvolume[None, ...].transpose()
        box_xyzc = box_czyx[:, ::-1]
        with Timer(f"Tensorstore: Writing {box_xyzc[:, :-1].tolist()} (XYZ)", logger):
            store = self.store(scale)
            fut = store[box_to_slicing(*box_xyzc)].write(vol_xyzc)
            fut.result()
