import os
import numpy as np

from confiddler import validate
from dvid_resource_manager.client import ResourceManagerClient
from intern.remote.boss.remote import BossRemote

from ..util import auto_retry, replace_default_entries
from . import VolumeServiceReader, GeometrySchema, SegmentationAdapters

BossServiceSchema = \
{
    "description": "Parameters to use the Boss as a source of voxel data",
    "type": "object",
    "required": ["project", "dataset", "volume-id", "change-stack-id"],
    "default": {},
    #"additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "host": {
            "description": "BOSS host url, e.g. api.bossdb.org",
            "type": "string"
        },
        "collection": {
            "description": "BOSS Collection",
            "type": "string",
        },
        "experiment": {
            "description": "BOSS Experiment",
            "type": "string"
        },
        "channel": {
            "description": "BOSS channel",
            "type": "string"
        }
    }
}

BossVolumeSchema = \
{
    "description": "Describes a segmentation volume from Boss.",
    "type": "object",
    "default": {},
#    "additionalProperties": False,
    "properties": {
        "boss": BossServiceSchema,
        "geometry": GeometrySchema,
        "adapters": SegmentationAdapters # Boss supports both segmentation and grayscale.
                                         # SegmentationAdapters is a superset of GrayscaleAdapters.
    }
}

class BossVolumeServiceReader(VolumeServiceReader):
    """
    A wrapper around the Boss client class that
    matches the VolumeServiceReader API.
    """

    def __init__(self, volume_config, resource_manager_client=None):
        validate(volume_config, BossVolumeSchema, inject_defaults=True)

        if resource_manager_client is None:
            # Dummy client
            resource_manager_client = ResourceManagerClient("", 0)

        try:
            token = os.environ["BOSS_TOKEN"]
        except KeyError:
            raise RuntimeError("You must define the BOSS_TOKEN environment variable to use BossVolumeService")

        self._boss = BossRemote(
                {
                        "protocol": "https",
                        "host": volume_config["boss"]["host"],
                        "token": token
                })

        self._channel = self._boss.get_channel(
                        volume_config["boss"]["channel"],
                        volume_config["boss"]["collection"],
                        volume_config["boss"]["experiment"],
                )

        block_width = volume_config["geometry"]["block-width"]
        if block_width == -1:
            # FIXME: I don't think that the Boss uses a cube for blocks internally...
            # specifically (x, y, z) dimensions are (512, 512, 16)
            block_width = 16

        preferred_message_shape_zyx = np.array( volume_config["geometry"]["message-block-shape"][::-1] )
        replace_default_entries(preferred_message_shape_zyx, [64, 64, 6400])

        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        if -1 in bounding_box_zyx.flat:
            raise RuntimeError("For BOSS volumes, you must explicity supply the entire bounding box in your config.")
        #replace_default_entries(bounding_box_zyx, self._boss.get_coordinate_frame....)

        assert  (bounding_box_zyx[0] >= self._boss_client.bounding_box[0]).all() \
            and (bounding_box_zyx[1] <= self._boss_client.bounding_box[1]).all(), \
            f"Specified bounding box ({bounding_box_zyx.tolist()}) extends outside the "\
            f"Boss volume geometry ({self._boss_client.bounding_box.tolist()})"

        available_scales = list(volume_config["geometry"]["available-scales"])

        # Store members
        self._bounding_box_zyx = bounding_box_zyx
        self._resource_manager_client = resource_manager_client
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._block_width = block_width
        self._available_scales = available_scales

        # Overwrite config entries that we might have modified
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()

    @property
    def dtype(self):
        return self._channel.datatype

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

    # Two-levels of auto-retry:
    # 1. Auto-retry up to three time for any reason.
    # 2. If that fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
    @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                predicate=lambda ex: '503' in str(ex.args[0]) or '504' in str(ex.args[0]))
    @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
    def get_subvolume(self, box, scale=0):
        req_bytes = 8 * np.prod(box[1] - box[0])
        with self._resource_manager_client.access_context('boss', True, 1, req_bytes):
            x_bounds = [box[0][2], box[1][2]]
            y_bounds = [box[0][1], box[1][1]]
            z_bounds = [box[0][0], box[1][0]]
            return self._boss.get_cutout(
                    self._channel,
                    scale,
                    x_bounds,
                    y_bounds,
                    z_bounds)
