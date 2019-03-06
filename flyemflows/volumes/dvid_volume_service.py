import socket
import logging

import numpy as np
import pandas as pd
from requests import HTTPError

from confiddler import validate
from dvid_resource_manager.client import ResourceManagerClient

from neuclease.util import Timer, choose_pyramid_depth, SparseBlockMask
from neuclease.dvid import ( fetch_repo_instances, fetch_instance_info, fetch_volume_box,
                             fetch_raw, post_raw, fetch_labelarray_voxels, post_labelmap_voxels,
                             update_extents, extend_list_value, create_voxel_instance, create_labelmap_instance,
                             fetch_sparsevol_coarse_via_labelindex, fetch_mapping )

from ..util import auto_retry, replace_default_entries
from . import GeometrySchema, VolumeServiceReader, VolumeServiceWriter, NewAxisOrderSchema, RescaleLevelSchema, LabelMapSchema

logger = logging.getLogger(__name__)

DvidInstanceCreationSettingsSchema = \
{
    "description": "Settings to use when creating a dvid volume (if necessary).\n"
                   "Note: To configure the block-width of the instance, specify the\n"
                   "      block-width setting in the volume's 'geometry' config section, below.\n",
    "type": "object",
    "default": {},
    "properties": {
        "versioned": {
            "description": "Whether or not the volume data should be versioned in DVID,\n"
                           "or if only a single copy of the volume will be stored\n"
                           "(in which case any update is visible in all nodes in the DAG).\n",
            "type": "boolean",
            "default": True
        },
        "tags": {
            "description": 'Optional "tags" to initialize the instance with, specified\n'
                           'as strings in the format key=value, e.g. "type=mask".\n',
            "type": "array",
            "items": {"type": "string"},
            "default": []
        },
        "enable-index": {
            "description": "(Labelmap instances only.)\n"
                           "Whether or not to support indexing on this labelmap instance.\n"
                           "Should usually be True, except for benchmarking purposes.\n"
                           "Note: This setting is distinct from the 'disable-indexing' setting \n"
                           "      in the volume service settings (below), which can be used to temporarily\n"
                           "      disable indexing updates that would normally be triggered after every post.\n",
            "type": "boolean",
            "default": True
        },
        "compression": {
            "description": "(Grayscale instances only -- labelmap instances are always compressed in a custom format.)\n"
                           "What type of compression is used by DVID to store this instance.\n"
                           "Choices: 'none' and 'jpeg'.\n",
            "type": "string",
            "enum": ["none", "jpeg"],
            "default": "none"
        },
        "max-scale": {
            "description": "The maximum pyramid scale to support in a labelmap instance,\n"
                           "or, in the case of grayscale volumes, the number of grayscale\n"
                           "instances to create, named with the convention '{name}_{scale}' (except scale 0).\n"
                           "If left unspecified (-1) an appropriate default will be chosen according\n"
                           "to the size of the volume geometry's bounding-box.\n"
                           "Note: For now, you are still requried to specify the 'available-scales'\n"
                           "setting in the volume geometry, too.\n",
            "type": "integer",
            "minValue": -1,
            "maxValue": 10, # Arbitrary max; larger than 10 is probably a sign that the user is just confused or made a typo.
            "default": -1
        },
        "voxel-size": {
            "description": "Voxel width, stored in DVID's metadata for the instance.",
            "type": "number",
            "default": 8.0
        },
        "voxel-units": {
            "description": "Physical units of the voxel-size specification.",
            "type": "string",
            "default": "nanometers"
        },
        "background": {
            "description": "(Grayscale only.) What pixel value should be considered 'background' (unpopulated voxels).\n",
            "type": "integer",
            "default": 0
        }
    }
}

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
        },
        "create-if-necessary": {
            "description": "Whether or not to create the instance if it doesn't already exist.\n"
                           "If you expect the instance to exist on the server already, leave this\n"
                           "set to False to avoid confusion in the case of typos, UUID mismatches, etc.\n"
                           "Note: When creating grayscale instances, ",
            "type": "boolean",
            "default": False
        },
        "creation-settings": DvidInstanceCreationSettingsSchema
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
                           "indexes will be sent by the client later on.\n"
                           "Note: This is different than the 'enable-index' creation setting, which specifies\n"
                           "      whether or not indexing will be available at all for the instance when it\n"
                           "      is created.",
            "type": "boolean",
            "default": False
        },
        "enable-downres": {
            "description": "When scale-0 data is posted, tell the server to update the low-resolution downsample pyramids.\n",
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

        assert ('segmentation-name' in volume_config["dvid"]) ^ ('grayscale-name' in volume_config["dvid"]), \
            "Config error: Specify either segmentation-name or grayscale-name (not both)"

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


            if not volume_config["dvid"]["create-if-necessary"]:
                existing_instances = fetch_repo_instances(self._server, self._uuid)
                if self._instance_name not in existing_instances:
                    raise RuntimeError("Instance '{self._instance_name}' does not exist in {self._server} / {self._uuid}."
                                       "Add 'create-if-necessary: true' to your config if you want it to be created.'")
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

        if "enable-downres" in volume_config["dvid"]:
            self.enable_downres = volume_config["dvid"]["enable-downres"]
        else:
            self.enable_downres = False

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

        ##
        ## Overwrite config entries that we might have modified
        ##
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()

        # TODO: Check the server for available scales and overwrite in the config?
        #volume_config["geometry"]["available-scales"] = [0]

        if volume_config["dvid"]["create-if-necessary"]:
            self._create_instance(volume_config)


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

    @property
    def resource_manager_client(self):
        return self._resource_manager_client


    def _create_instance(self, volume_config):
        if 'segmentation-name' in volume_config["dvid"]:
            self._create_labelmap_instance(volume_config)
        if 'grayscale-name' in volume_config["dvid"]:
            self._create_grayscale_instances(volume_config) 
            

    def _create_labelmap_instance(self, volume_config):
        if self.instance_name in fetch_repo_instances(self.server, self.uuid):
            logger.info(f"'{self.instance_name}' already exists, skipping creation")
            return

        settings = volume_config["dvid"]["creation-settings"]
        block_width = volume_config["geometry"]["block-width"]

        pyramid_depth = settings["max-scale"]
        if pyramid_depth == -1:
            pyramid_depth = choose_pyramid_depth(self.bounding_box_zyx, 512)

        if settings["compression"] != DvidInstanceCreationSettingsSchema["properties"]["compression"]["default"]:
            raise RuntimeError("Alternative compression methods are not permitted on labelmap instances. "
                               "Please remove the 'compression' setting from your config.")

        if settings["background"] != 0:
            raise RuntimeError("Labelmap instances do not support custom background values. "
                               "Please remove 'background' from your config.")

        create_labelmap_instance( self.server,
                                  self.uuid,
                                  self.instance_name,
                                  settings["versioned"],
                                  settings["tags"],
                                  block_width,
                                  settings["voxel-size"],
                                  settings["voxel-units"],
                                  settings["enable-index"],
                                  pyramid_depth )


    def _create_grayscale_instances(self, volume_config):
        settings = volume_config["dvid"]["creation-settings"]
        
        block_width = volume_config["geometry"]["block-width"]

        pyramid_depth = settings["max-scale"]
        if pyramid_depth == -1:
            pyramid_depth = choose_pyramid_depth(self.bounding_box_zyx, 512)

        repo_instances = fetch_repo_instances(self.server, self.uuid)

        # Bottom level of pyramid is listed as neuroglancer-compatible
        extend_list_value(self.server, self.uuid, '.meta', 'neuroglancer', [self.instance_name])
        
        for scale in range(pyramid_depth+1):
            scaled_output_box_zyx = self.bounding_box_zyx // 2**scale # round down
    
            if scale == 0:
                scaled_instance_name = self.instance_name
            else:
                scaled_instance_name = f"{self.instance_name}_{scale}"
    
            if scaled_instance_name in repo_instances:
                logger.info(f"'{scaled_instance_name}' already exists, skipping creation")
            else:
                create_voxel_instance( self.server,
                                       self.uuid,
                                       scaled_instance_name,
                                       'uint8blk',
                                       settings["versioned"],
                                       settings["compression"],
                                       settings["tags"],
                                       block_width,
                                       settings["voxel-size"],
                                       settings["voxel-units"],
                                       settings["background"] )
    
            update_extents( self.server, self.uuid, scaled_instance_name, scaled_output_box_zyx )

            # Higher-levels of the pyramid should not appear in the DVID console.
            extend_list_value(self.server, self.uuid, '.meta', 'restrictions', [scaled_instance_name])


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
                                                         format='lazy-array' )
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
        
        assert (not self.enable_downres) or (scale == 0), \
            "When using enable-downres, you can only write scale-0 data."
        
        try:
            # Labelarray data can be posted very efficiently if the request is block-aligned
            if self._instance_type in ('labelarray', 'labelmap') and is_block_aligned:
                with self._resource_manager_client.access_context(self._server, True, 1, req_bytes):
                    post_labelmap_voxels( self._server, self._uuid, instance_name, offset_zyx, subvolume, scale,
                                            self.enable_downres, self.disable_indexing, throttle )
            else:
                assert not self.enable_downres, \
                    "Can't use enable-downres: You are attempting to post non-block-aligned data."

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


    def sparse_block_mask_for_labels(self, labels):
        """
        Determine which bricks (each with our ``preferred_message_shape``
        would need to be accessed download all data for the given labels,
        and return the result as a ``SparseBlockMask`` object.
        
        This function uses a dask to fetch the coarse sparsevols in parallel.
        The sparsevols are extracted directly from the labelindex.
        If the ``self.supervoxels`` is True, the labels are grouped
        by body before fetching the labelindexes,
        to avoid fetching the same labelindexes more than once.
        (The implementation of ``fetch_sparsevol_coarse_via_labelindex()``
        avoids fetching duplicate labelindexes when handling supervoxels
        that belong to the same body.)
        """
        labels = set(labels)
        is_supervoxels = self.supervoxels
        brick_shape = self.preferred_message_shape
        blocks_per_brick = brick_shape // self.block_width

        if is_supervoxels:
            # Group by body
            mapping = fetch_mapping(*self.instance_triple, list(labels))
            label_groups = list(mapping.reset_index().groupby('body').agg({'sv': list})['sv'])
        else:
            # Each body is its own group
            label_groups = [[label] for label in labels]

        def fetch_coarse_and_divide(label_group):
            """
            Fetch the coarse sparsevol for the label and return a pair of
            (label, coords), where coords is a record array
            with columns [z,y,x,label], and z,y,x have been converted to
            brick indices (not block indicies).
            """
            try:
                mgr = self.resource_manager_client
                with mgr.access_context(self.server, True, 1, 1):
                    coords = fetch_sparsevol_coarse_via_labelindex(*self.instance_triple, label_group, is_supervoxels)
                    if len(coords) == 0:
                        return (label_group, None)
                    coords //= blocks_per_brick
                    coords = pd.DataFrame(coords, columns=['z', 'y', 'x'], dtype=np.int32).drop_duplicates()
                return (label_group, coords.to_records(index=False))
            except HTTPError as ex:
                if (ex.response is not None and ex.response.status_code == 404):
                    return (label_group, None)
                raise
            except RuntimeError as ex:
                if 'does not map to any body' in str(ex):
                    return (label_group, None)
                raise

        with Timer(f"Fetching coarse sparsevols for {len(labels)} labels ({len(label_groups)} groups)", logger=logger):
            import dask.bag as db
            label_groups = db.from_sequence(label_groups)
            groups_and_coords = label_groups.map(fetch_coarse_and_divide).compute()

        bad_labels = []
        for label_group, coords in groups_and_coords:
            if coords is None:
                bad_labels.extend(label_group)
    
        if bad_labels:
            name = 'sv' if is_supervoxels else 'body'
            pd.Series(bad_labels, name=name).to_csv('labels-without-sparsevols.csv', index=False, header=True)
            if len(bad_labels) < 100:
                msg = f"Could not obtain coarse sparsevol for {len(bad_labels)} labels: {bad_labels}"
            else:
                msg = f"Could not obtain coarse sparsevol for {len(bad_labels)} labels. See labels-without-sparsevols.csv"

            logger.warning(msg)

            # Drop null groups
            groups_and_coords = list( filter(lambda k_v: k_v[1] is not None, groups_and_coords) )

        if len(groups_and_coords) == 0:
            empty_box = np.array([self.bounding_box_zyx[0], self.bounding_box_zyx[0]])
            empty_mask = np.zeros((0,)*len(empty_box[0]), dtype=bool)
            return SparseBlockMask(empty_mask, empty_box, brick_shape)
            
        all_coords = np.concatenate( [kv[1] for kv in groups_and_coords] )
        coords_df = pd.DataFrame( all_coords )
        coords_df.drop_duplicates(['z', 'y', 'x'], inplace=True)
        coords = coords_df[['z', 'y', 'x']].values
        
        min_coord = coords.min(axis=0)
        max_coord = coords.max(axis=0)
        mask_box = np.array((min_coord, 1+max_coord))
        mask_shape = mask_box[1] - mask_box[0]
        mask = np.zeros(mask_shape, dtype=bool)

        coords -= mask_box[0]
        mask[(*coords.transpose(),)] = True

        sbm = SparseBlockMask(mask, brick_shape*mask_box, brick_shape)
        return sbm

