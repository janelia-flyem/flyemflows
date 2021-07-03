import copy
import pickle
import numpy as np

from confiddler import validate, emit_defaults
from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import box_to_slicing

from ..util import auto_retry, replace_default_entries
from . import VolumeServiceReader, GeometrySchema, SegmentationAdapters

TensorStoreSpecSchema = \
{
    "description": "TensorStore Spec",
    "type": "object",
    "default": {},
    "required": ["driver", "kvstore", "path"],
    # "additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "driver": {
            "description": "type of volume",
            "type": "string",
            "enum": ["neuroglancer_precomputed"],
            "default": "neuroglancer_precomputed"
        },
        "kvstore": {
            "type": "object",
            "default": {},
            "properties": {
                "driver": {
                    "type": "string",
                    "enum": ["gcs"],
                    "default": "gcs"
                },
                "bucket": {
                    "description": "bucket name",
                    "type": "string"
                    # no default
                }
            }
        },
        "path": {
            "description": "Path within the bucket",
            "type": "string"
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
        },

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
    "description": "TensorStore Service settings",
    "type": "object",
    "default": {},
    "required": ["spec", "context"],
    # "additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "spec": TensorStoreSpecSchema,
        "context": TensorStoreContextSchema,
        "reinitialize-via": {
            "type": "string",
            "enum": ["unpickle", "reopen"],
            "default": "unpickle"
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


class TensorStoreVolumeServiceReader(VolumeServiceReader):
    """
    A wrapper around the TensorStore API to implement the VolumeServiceReader API.
    """

    @classmethod
    def default_config(cls, ng_src_url=None):
        """
        Return the default config for a TensorStoreVolumeReader.
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
                assert parts[-2] == 'gs'

            bucket, *path_parts = parts[-1].split('/')
            cfg['tensorstore']['spec']['kvstore']['bucket'] = bucket
            cfg['tensorstore']['spec']['path'] = '/'.join(path_parts)

        return cfg

    @classmethod
    def from_url(cls, ng_src_url):
        """
        Construct a TensorStoreVolumeServiceReader from a neuroglancer source url,
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

        self._stores = {}
        store = self.store(0)
        spec = store.spec()

        block_width = volume_config["geometry"]["block-width"]
        if block_width == -1:
            block_width = spec.to_json()['scale_metadata']['chunk_size'][0]

        preferred_message_shape_zyx = np.array( volume_config["geometry"]["message-block-shape"][::-1] )
        replace_default_entries(preferred_message_shape_zyx, [256, 256, 256])

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
        self._dtype = spec.dtype.numpy_dtype
        self._block_width = block_width
        self._bounding_box_zyx = bounding_box_zyx
        self._resource_manager_client = resource_manager_client
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._available_scales = available_scales
        self._reinitialize_via = volume_config["tensorstore"]["reinitialize-via"]

        # Overwrite config entries that we might have modified
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()

    def store(self, scale):
        try:
            return self._stores[scale]
        except KeyError:
            import tensorstore as ts
            spec = copy.copy(self.volume_config['tensorstore']['spec'])
            spec['scale_index'] = scale

            context = self.volume_config['tensorstore']['context']
            store = ts.open(spec, read=True, write=False, context=ts.Context(context)).result()
            self._stores[scale] = store
            return store

    def __getstate__(self):
        """
        Pickle representation.
        """
        d = self.__dict__.copy()

        # Apparently unpickling can fail intermittently due to network issues, so we
        # pickle/unpickle it explicitly so we can control the process and retry if necessary.
        d['_stores'] = pickle.dumps(d['_stores'])

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

        self.__dict__ = d

    @property
    def dtype(self):
        return self._dtype

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape_zyx

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
    # If a failure is not 503, restart it up to three times, with a short delay.
    # If the request fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
    @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                predicate=lambda ex: '503' in str(ex.args[0]) or '504' in str(ex.args[0]))
    @auto_retry(3, pause_between_tries=30.0, logging_name=__name__,
                predicate=lambda ex: '503' not in str(ex.args[0]) and '504' not in str(ex.args[0]))
    def get_subvolume(self, box_zyx, scale=0):
        box_zyx = np.asarray(box_zyx)
        req_bytes = 8 * np.prod(box_zyx[1] - box_zyx[0])
        try:
            resource_name = self.volume_config['tensorstore']['spec']['kvstore']['bucket']
        except KeyError:
            resource_name = self.volume_config['tensorstore']['spec']['path']

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
            return vol_zyx
