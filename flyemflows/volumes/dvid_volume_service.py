import socket
import logging
import numpy as np
from requests import HTTPError

from confiddler import validate
from dvid_resource_manager.client import ResourceManagerClient

from neuclease.util import boxes_from_grid, box_to_slicing
from neuclease.dvid import fetch_instance_info, fetch_volume_box, fetch_raw, post_raw, fetch_labelarray_voxels, post_labelarray_blocks

from ..util import auto_retry, replace_default_entries
from . import GeometrySchema, VolumeServiceReader, VolumeServiceWriter, NewAxisOrderSchema, RescaleLevelSchema, LabelMapSchema

logger = logging.getLogger(__name__)

DvidServiceSchema = \
{
    "description": "Parameters specify a DVID node",
    "type": "object",
    "required": ["server", "uuid"],

    "default": {},
    "properties": {
        "server": {
            "description": "location of DVID server to READ.",
            "type": "string",
        },
        "uuid": {
            "description": "version node for READING segmentation",
            "type": "string"
        }
    }
}

DvidGrayscaleServiceSchema = \
{
    "description": "Parameters specify a source of grayscale data from DVID",
    "type": "object",

    "allOf": [DvidServiceSchema],

    "required": DvidServiceSchema["required"] + ["grayscale-name"],
    "default": {},
    "properties": {
        "grayscale-name": {
            "description": "The grayscale instance to read/write from/to.\n"
                           "Instance must be grayscale (uint8blk).",
            "type": "string",
            "minLength": 1
        },
        "compression": {
            "description": "What type of compression is used to store this instance.\n"
                           "(Only used when the instance is created for the first time.)\n"
                           "Choices: 'raw' and 'jpeg'.\n",
            "type": "string",
            "enum": ["raw", "jpeg"],
            "default": "raw"
        }
    }
}

DvidSegmentationServiceSchema = \
{
    "description": "Parameters specify a source of segmentation data from DVID",
    "type": "object",

    "allOf": [DvidServiceSchema],

    "required": DvidServiceSchema["required"] + ["segmentation-name"],
    "default": {},
    #"additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "segmentation-name": {
            "description": "The labels instance to read/write from. \n"
                           "Instance may be either googlevoxels, labelblk, or labelarray.",
            "type": "string",
            "minLength": 1
        },
        "supervoxels": {
            "description": "Whether or not to read/write supervoxels from the labelmap instance, not agglomerated labels.\n"
                           "Applies to labelmap instances only.",
            "type": "boolean",
            "default": False
        },
        "disable-indexing": {
            "description": "Tell the server not to update the label index after POST blocks.\n"
                           "Useful during initial volume ingestion, in which label\n"
                           "indexes will be sent by the client later on.\n",
            "type": "boolean",
            "default": False
        }
    }
}

DvidGenericVolumeSchema = \
{
    "description": "Schema for a generic dvid volume",
    "type": "object",
    "default": {},
    #"additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "dvid": { "oneOf": [DvidGrayscaleServiceSchema, DvidSegmentationServiceSchema] },
        "geometry": GeometrySchema,
        "transpose-axes": NewAxisOrderSchema,
        "rescale-level": RescaleLevelSchema,
        "apply-labelmap": LabelMapSchema
    }
}

DvidSegmentationVolumeSchema = \
{
    "description": "Schema for a segmentation dvid volume", # (for when a generic SegmentationVolumeSchema won't suffice)
    "type": "object",
    "default": {},
    #"additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "dvid": DvidSegmentationServiceSchema,
        "geometry": GeometrySchema,
        "transpose-axes": NewAxisOrderSchema,
        "rescale-level": RescaleLevelSchema,
        "apply-labelmap": LabelMapSchema
    }
}


class DvidVolumeService(VolumeServiceReader, VolumeServiceWriter):

    def __init__(self, volume_config, resource_manager_client=None):
        validate(volume_config, DvidGenericVolumeSchema, inject_defaults=True)
        
        assert 'apply-labelmap' not in volume_config["dvid"].keys(), \
            "The apply-labelmap section should be parallel to 'dvid' and 'geometry', not nested within the 'dvid' section!"

        ##
        ## server, uuid
        ##
        if not volume_config["dvid"]["server"].startswith('http://'):
            volume_config["dvid"]["server"] = 'http://' + volume_config["dvid"]["server"]
        
        self._server = volume_config["dvid"]["server"]
        self._uuid = volume_config["dvid"]["uuid"]

        ##
        ## instance, dtype, etc.
        ##

        config_block_width = volume_config["geometry"]["block-width"]

        if "segmentation-name" in volume_config["dvid"]:
            self._instance_name = volume_config["dvid"]["segmentation-name"]
            self._dtype = np.uint64
        elif "grayscale-name" in volume_config["dvid"]:
            self._instance_name = volume_config["dvid"]["grayscale-name"]
            self._dtype = np.uint8
            
        self._dtype_nbytes = np.dtype(self._dtype).type().nbytes

        try:
            instance_info = fetch_instance_info(self._server, self._uuid, self._instance_name)
        except HTTPError as ex:
            if ex.response.status_code != 400:
                raise

            # Instance doesn't exist yet -- we are going to create it.
            if "segmentation-name" in volume_config["dvid"]:
                self._instance_type = 'labelmap' # get_voxels doesn't really care if it's labelarray or labelmap...
                self._is_labels = True
            else:
                self._instance_type = 'uint8blk'
                self._is_labels = False
            
            block_width = config_block_width
        else:
            self._instance_type = instance_info["Base"]["TypeName"]
            self._is_labels = self._instance_type in ('labelblk', 'labelarray', 'labelmap')
            if self._instance_type == "googlevoxels" and instance_info["Extended"]["Scales"][0]["channelType"] == "UINT64":
                self._is_labels = True

            bs_x, bs_y, bs_z = instance_info["Extended"]["BlockSize"]
            assert (bs_x == bs_y == bs_z), "Expected blocks to be cubes."
            block_width = bs_x


        if "disable-indexing" in volume_config["dvid"]:
            self.disable_indexing = volume_config["dvid"]["disable-indexing"]
        else:
            self.disable_indexing = False

        # Whether or not to read the supervoxels from the labelmap instance instead of agglomerated labels.
        self.supervoxels = ("supervoxels" in volume_config["dvid"]) and (volume_config["dvid"]["supervoxels"])


        ##
        ## default block width
        ##
        assert config_block_width in (-1, block_width), \
            f"DVID volume block-width ({config_block_width}) from config does not match server metadata ({block_width})"
        if block_width == -1:
            # No block-width specified; choose default
            block_width = 64


        ##
        ## bounding-box
        ##
        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        try:
            stored_extents = fetch_volume_box(self._server, self.uuid, self._instance_name)
        except HTTPError:
            assert -1 not in bounding_box_zyx.flat[:], \
                f"Instance '{self._instance_name}' does not yet exist on the server, "\
                "so your volume_config must specify explicit values for bounding-box"
        else:
            replace_default_entries(bounding_box_zyx, stored_extents)

        ##
        ## message-block-shape
        ##
        preferred_message_shape_zyx = np.array( volume_config["geometry"]["message-block-shape"][::-1] )
        replace_default_entries(preferred_message_shape_zyx, [block_width, block_width, 100*block_width])

        ##
        ## available-scales
        ##
        available_scales = list(volume_config["geometry"]["available-scales"])

        ##
        ## resource_manager_client
        ##
        if resource_manager_client is None:
            # Dummy client
            resource_manager_client = ResourceManagerClient("", 0)

        ##
        ## Store members
        ##
        self._resource_manager_client = resource_manager_client
        self._block_width = block_width
        self._bounding_box_zyx = bounding_box_zyx
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._available_scales = available_scales

        # Memoized in the node_service property.
        self._node_service = None

        ##
        ## Overwrite config entries that we might have modified
        ##
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()

        # TODO: Check the server for available scales and overwrite in the config?
        #volume_config["geometry"]["available-scales"] = [0]

    @property
    def server(self):
        return self._server

    @property
    def uuid(self):
        return self._uuid
    
    @property
    def instance_name(self):
        return self._instance_name

    @property
    def instance_triple(self):
        return (self.server, self.uuid, self.instance_name)

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

    # Two-levels of auto-retry:
    # 1. Auto-retry up to three time for any reason.
    # 2. If that fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
    @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                predicate=lambda ex: '503' in str(ex.args[0]) or '504' in str(ex.args[0]))
    @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
    def get_subvolume(self, box_zyx, scale=0):
        req_bytes = self._dtype_nbytes * np.prod(box_zyx[1] - box_zyx[0])
        throttle = (self._resource_manager_client.server_ip == "")

        instance_name = self._instance_name
        if self._instance_type.endswith('blk') and scale > 0:
            # Grayscale multi-scale is achieved via multiple instances
            instance_name = f"{instance_name}_{scale}"
            scale = 0

        try:
            if self._instance_type in ('labelarray', 'labelmap'):
                # Obtain permission from the resource manager while fetching the compressed data,
                # but release the resource token before inflating the data.
                with self._resource_manager_client.access_context(self._server, True, 1, req_bytes):
                    vol_proxy = fetch_labelarray_voxels( self._server,
                                                         self._uuid,
                                                         instance_name,
                                                         box_zyx,
                                                         scale,
                                                         throttle,
                                                         supervoxels=self.supervoxels,
                                                         inflate=False )
                # Inflate
                return vol_proxy()
            else:
                with self._resource_manager_client.access_context(self._server, True, 1, req_bytes):
                    return fetch_raw(self._server, self._uuid, instance_name, box_zyx, throttle)

        except Exception as ex:
            # In certain cluster scenarios, the 'raise ... from ex' traceback doesn't get fully transmitted to the driver.
            import traceback, io
            sio = io.StringIO()
            traceback.print_exc(file=sio)
            logger.log(logging.ERROR, sio.getvalue() )

            host = socket.gethostname()
            msg = f"Host {host}: Failed to fetch subvolume: box_zyx = {box_zyx.tolist()}"
            raise RuntimeError(msg) from ex
        
    # Two-levels of auto-retry:
    # 1. Auto-retry up to three time for any reason.
    # 2. If that fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
    @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                predicate=lambda ex: '503' in str(ex.args[0]) or '504' in str(ex.args[0]))
    @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
    def write_subvolume(self, subvolume, offset_zyx, scale):
        req_bytes = self._dtype_nbytes * np.prod(subvolume.shape)
        throttle = (self._resource_manager_client.server_ip == "")

        offset_zyx = np.asarray(offset_zyx)
        shape_zyx = np.asarray(subvolume.shape)
        box_zyx = np.array([offset_zyx,
                            offset_zyx + shape_zyx])
        
        instance_name = self._instance_name

        if self._instance_type.endswith('blk') and scale > 0:
            # Grayscale multi-scale is achieved via multiple instances
            instance_name = f"{instance_name}_{scale}"
            scale = 0

        if self._instance_type == 'labelmap':
            assert self.supervoxels, "You cannot post data to a labelmap instance unless you specify 'supervoxels: true' in your config."

        is_block_aligned = (box_zyx % self.block_width == 0).all()
        
        try:
            # Labelarray data can be posted very efficiently if the request is block-aligned
            if self._instance_type in ('labelarray', 'labelmap') and is_block_aligned:
                block_boxes = list(boxes_from_grid(box_zyx, self.block_width))
                corners = [box[0] for box in block_boxes]
                blocks = (subvolume[box_to_slicing(*(box - offset_zyx))] for box in block_boxes)

                with self._resource_manager_client.access_context(self._server, True, 1, req_bytes):
                    post_labelarray_blocks( self._server, self._uuid, instance_name, corners, blocks, scale,
                                            downres=False, noindexing=self.disable_indexing, throttle=throttle )
            else:
                with self._resource_manager_client.access_context(self._server, True, 1, req_bytes):
                    post_raw( self._server, self._uuid, instance_name, offset_zyx, subvolume,
                              throttle=throttle, mutate=not self.disable_indexing )

        except Exception as ex:
            # In certain cluster scenarios, the 'raise ... from ex' traceback doesn't get fully transmitted to the driver.
            import traceback, io
            sio = io.StringIO()
            traceback.print_exc(file=sio)
            logger.log(logging.ERROR, sio.getvalue() )

            host = socket.gethostname()
            msg = f"Host {host}: Failed to write subvolume: offset_zyx = {offset_zyx.tolist()}, shape = {subvolume.shape}"
            raise RuntimeError(msg) from ex

