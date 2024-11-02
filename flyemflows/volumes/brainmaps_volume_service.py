import numpy as np

from confiddler import validate
from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import boxes_from_grid, box_to_slicing

from ..util import auto_retry, replace_default_entries
from . import VolumeServiceReader, GeometrySchema, SegmentationAdapters
from .brainmaps_volume import BrainMapsVolume

BrainMapsServiceSchema = \
{
    "description": "Parameters to use Google BrainMaps as a source of voxel data",
    "type": "object",
    "required": ["project", "dataset", "volume-id", "change-stack-id"],
    "default": {},
    #"additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "project": {
            "description": "Project ID",
            "oneOf": [{"type": "string"}, {"type": "integer"}]
        },
        "dataset": {
            "description": "Dataset identifier",
            "type": "string"
        },
        "volume-id": {
            "description": "Volume ID",
            "type": "string"
        },
        "change-stack-id": {
            "description": "Change Stack ID. Specifies a set of changes to apple on top of the volume\n"
                           "(e.g. a set of agglomeration steps).",
            "type": "string",
            "default": ""
        },
        "use-gzip": {
            "description": "Whether or not to use gzip transfer encoding (on top of the snappy compression)",
            "type": "boolean",
            "default": True
        },
        "fetch-blockwise": {
            "description": "When fetching each subvolume, split the fetch across multiple requests,\n"
                           "according to the block-width specifed in the volume geometry metadata.\n"
                           "(If this setting is enabled and you don't set the block-width, it defaults to 64.)\n",
            "type": "boolean",
            "default": False
        }
    }
}

BrainMapsVolumeSchema = \
{
    "description": "Describes a segmentation volume from BrainMaps.",
    "type": "object",
    "default": {},
#    "additionalProperties": False,
    "properties": {
        "brainmaps": BrainMapsServiceSchema,
        "geometry": GeometrySchema,
        "adapters": SegmentationAdapters # Brainmaps supports both segmentation and grayscale.
                                         # SegmentationAdapters is a superset of GrayscaleAdapters.
    }
}


class BrainMapsVolumeServiceReader(VolumeServiceReader):
    """
    A wrapper around the BrainMaps client class that
    matches the VolumeServiceReader API.
    """

    def __init__(self, volume_config, resource_manager_client=None):
        validate(volume_config, BrainMapsVolumeSchema, inject_defaults=True)

        if resource_manager_client is None:
            # Dummy client
            resource_manager_client = ResourceManagerClient("", 0)

        self._brainmaps_client = BrainMapsVolume( str(volume_config["brainmaps"]["project"]),
                                                  volume_config["brainmaps"]["dataset"],
                                                  volume_config["brainmaps"]["volume-id"],
                                                  volume_config["brainmaps"]["change-stack-id"],
                                                  dtype=None,
                                                  use_gzip=volume_config["brainmaps"]["use-gzip"] )

        # Force client to fetch dtype now, so it isn't fetched after pickling.
        self._brainmaps_client.dtype

        block_width = volume_config["geometry"]["block-width"]
        if block_width == -1:
            # FIXME: I don't actually know what BrainMap's internal block size is...
            block_width = 64

        preferred_message_shape_zyx = np.array( volume_config["geometry"]["message-block-shape"][::-1] )
        replace_default_entries(preferred_message_shape_zyx, [64, 64, 6400])

        preferred_grid_offset_zyx = np.array( volume_config["geometry"]["message-grid-offset"][::-1] )

        uncropped_bounding_box_zyx = np.array(volume_config["geometry"]["uncropped-bounding-box"])[:,::-1]
        replace_default_entries(uncropped_bounding_box_zyx, self._brainmaps_client.bounding_box)

        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        replace_default_entries(bounding_box_zyx, uncropped_bounding_box_zyx)

        assert  (bounding_box_zyx[0] >= self._brainmaps_client.bounding_box[0]).all() \
            and (bounding_box_zyx[1] <= self._brainmaps_client.bounding_box[1]).all(), \
            f"Specified bounding box ({bounding_box_zyx.tolist()}) extends outside the "\
            f"BrainMaps volume geometry ({self._brainmaps_client.bounding_box.tolist()})"

        available_scales = list(volume_config["geometry"]["available-scales"])
        fetch_blockwise = volume_config["brainmaps"]["fetch-blockwise"]

        # Store members
        self._uncropped_bounding_box_zyx = uncropped_bounding_box_zyx
        self._bounding_box_zyx = bounding_box_zyx
        self._resource_manager_client = resource_manager_client
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._preferred_grid_offset_zyx = preferred_grid_offset_zyx
        self._block_width = block_width
        self._available_scales = available_scales
        self._fetch_blockwise = fetch_blockwise

        # Overwrite config entries that we might have modified
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["uncropped-bounding-box"] = self._uncropped_bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()
        volume_config["geometry"]["message-grid-offset"] = self._preferred_grid_offset_zyx[::-1].tolist()

    @property
    def dtype(self):
        return self._brainmaps_client.dtype

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
    def uncropped_bounding_box_zyx(self):
        return self._uncropped_bounding_box_zyx

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
    def get_subvolume(self, box, scale=0):
        req_bytes = 8 * np.prod(box[1] - box[0])
        with self._resource_manager_client.access_context('brainmaps', True, 1, req_bytes):
            if not self._fetch_blockwise:
                return self._brainmaps_client.get_subvolume(box, scale)
            else:
                block_shape = 3*(self._block_width,)
                subvol = np.zeros(box[1] - box[0], self.dtype)
                for block_box in boxes_from_grid(box, block_shape, clipped=True):
                    block = self._brainmaps_client.get_subvolume(block_box, scale)
                    outbox = block_box - box[0]
                    subvol[box_to_slicing(*outbox)] = block
                return subvol
