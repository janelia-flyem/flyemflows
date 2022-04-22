import os
import copy
import logging
import pickle

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease.dvid import (fetch_repo_instances, create_tarsupervoxel_instance,
                            create_instance, is_locked, post_load, post_keyvalues, fetch_exists, fetch_key,
                            fetch_server_info, fetch_mapping, resolve_ref)
from neuclease.util import meshes_from_volume

from dvid_resource_manager.client import ResourceManagerClient

from ..util.dask_util import persist_and_execute
from .util.config_helpers import BodyListSchema, load_body_list
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService
from ..brick import BrickWall
from . import Workflow

logger = logging.getLogger(__name__)


class GridMeshes(Workflow):
    """
    Produce meshes for a 'gridded' supervoxel segmentation, i.e. a segmentation
    in which each supervoxel is restricted to a single 'grid' cube, thus making
    it easy to process supervoxels with local analyses only.  No 'reduce' step
    is necessary to aggregate results from different grid locations.
    """
    GenericDvidInstanceSchema = \
    {
        "description": "Parameters to specify a generic dvid instance (server/uuid/instance).\n"
                       "Omitted values will be copied from the input, or given default values.",
        "type": "object",
        "required": ["server", "uuid"],

        # "default": {}, # Must not have default. (Appears below in a 'oneOf' context.)
        "additionalProperties": False,
        "properties": {
            "server": {
                "description": "location of DVID server to READ.",
                "type": "string",
                "default": ""
            },
            "uuid": {
                "description": "version node from dvid",
                "type": "string",
                "default": ""
            },
            "instance": {
                "description": "Name of the instance to create",
                "type": "string"
            },
            "sync-to": {
                "description": "When creating a tarsupervoxels instance, it should be sync'd to a labelmap instance.\n"
                               "Give the instance name here.",
                "type": "string",
                "default": ""
            },
            "create-if-necessary": {
                "description": "Whether or not to create the instance if it doesn't already exist.\n"
                               "If you expect the instance to exist on the server already, leave this\n"
                               "set to False to avoid confusion in the case of typos, UUID mismatches, etc.\n",
                "type": "boolean",
                "default": False
            },
        }
    }

    TarsupervoxelsOutputSchema = \
    {
        "additionalProperties": False,
        "properties": {
            "tarsupervoxels": GenericDvidInstanceSchema
        }
    }

    KeyvalueOutputSchema = \
    {
        "additionalProperties": False,
        "properties": {
            "keyvalue": GenericDvidInstanceSchema
        }
    }

    DirectoryOutputSchema = \
    {
        "additionalProperties": False,
        "properties": {
            "directory": {
                "description": "Directory to write supervoxel meshes into.",
                "type": "string",
                #"default": "" # Must not have default. (Appears below in a 'oneOf' context.)
            }
        }
    }

    MeshParametersSchema = \
    {
        "default": {},
        "properties": {
            "smoothing": {
                "description": "How many iterations of smoothing to apply to each mesh before decimation.",
                "type": "integer",
                "default": 0
            },
            "decimation": {
                "description": "Mesh decimation aims to reduce the number of \n"
                               "mesh vertices in the mesh to a fraction of the original mesh. \n"
                               "To disable decimation, use 1.0.\n",
                "type": "number",
                "minimum": 0.0000001,
                "maximum": 1.0, # 1.0 == disable
                "default": 1.0
            },
            "compute-normals": {
                "description": "Compute vertex normals and include them in the uploaded results.",
                "type": "boolean",
                "default": False
            }
        }
    }
    GridMeshesOptionsSchema = \
    {
        "type": "object",
        "description": "Settings specific to the GridMeshes workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "subset-supervoxels": {
                **BodyListSchema,
                "description": "List of supervoxel IDs to process, or a path to a CSV file with the list.\n"
                               "NOTE: If you're using a non-labelmap source (e.g. HDF5, etc.),\n"
                               "      it is considered supervoxel data.\n",
            },
            "subset-bodies": {
                **BodyListSchema,
                "description": "List of body IDs to process, a path to a CSV file with the list,\n"
                               "or a timestamp (e.g. '2018-11-22 17:34:00') which will be used with \n"
                               "the kafka log to determine bodies that have changed (since the given time).\n"
                               "NOTE: If you're using a non-labelmap source (e.g. HDF5, etc.), \n"
                               "      it is considered supervoxel data.\n"
                               "      This config setting can only be used when using a labelmap source.\n",
            },
            "minimum-supervoxel-size": {
                "type": "number",
                "default": 0
            },
            "maximum-supervoxel-size": {
                "type": "number",
                "default": 1e12
            },
            "mesh-parameters": MeshParametersSchema,

            "rescale-before-write": {
                "description": "How much to rescale the meshes before writing to DVID.\n"
                               "Specified as a multiplier, not power-of-2 'scale'.\n"
                               "For anisotropic scaling, provide a list of [X,Y,Z] scaling multipliers.\n",
                "oneOf": [{"type": "number"},
                          {"type": "array", "items": {"type": "number"}}],
                "default": 1.0
            },
            "format": {
                "description": "Format to save the meshes in. Either obj, drc, or ngmesh.\n"
                               "Note: Neuroglancer meshes are expected to be saved using nm units,\n"
                               "      but the meshes from this workflow will use voxel units by default.\n"
                               "      When using the ngmesh format, use rescale-before-write to multiply all mesh coordinates accordingly.\n"
                               "      For example, use rescale-before-write: [16,16,16] if your volume is stored at is at 8nm resolution,\n"
                               "      and you are fetching from scale 1 (2x).\n",
                "type": "string",
                "enum": ["obj",     # Wavefront OBJ (.obj)
                         "drc",     # Draco (compressed) (.drc)
                         "ngmesh"], # "neuroglancer mesh" format -- a custom binary format.  See note above about scaling.
                "default": "obj"
            },
            "skip-existing": {
                "description": "Do not generate meshes for meshes that already exist in the output location.\n",
                "type": "boolean",
                "default": False
            },
            "sparse-block-mask": {
                "description": "Optionally provide a mask which limits the set of bricks to be processed.\n"
                               "If you already have a map of where the valid data is, you can provide a\n"
                               "pickled SparseBlockMask here.\n",
                "type": "string",
                "default": ""
            },

        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,

        "output": {
            "oneOf": [
                DirectoryOutputSchema,
                KeyvalueOutputSchema,
                TarsupervoxelsOutputSchema
            ],
            "default": {"directory": "meshes"}
        },

        "gridmeshes": GridMeshesOptionsSchema
    })

    @classmethod
    def schema(cls):
        return GridMeshes.Schema

    def execute(self):
        """
        """
        self._sanitize_config()
        subset_supervoxels_, subset_bodies = self._init_input_service()
        self._prepare_output()

        config = self.config
        input_service = self.input_service
        resource_mgr = self.mgr_client

        def process_brick(brick):
            subset_supervoxels = subset_supervoxels_
            options = config["gridmeshes"]
            all_svs = None
            if len(subset_bodies):
                all_svs = pd.unique(brick.volume.ravel())
                mappings = fetch_mapping(*input_service.base_service.instance_triple, all_svs, as_series=True)
                subset_supervoxels = mappings[mappings.isin(subset_bodies)].index

            if len(subset_supervoxels) == 0:
                subset_supervoxels = None

            if options["skip-existing"]:
                if subset_supervoxels is None:
                    subset_supervoxels = all_svs

                if subset_supervoxels is None:
                    subset_supervoxels = pd.unique(brick.volume.ravel())

                existing_svs = _determine_existing(config, subset_supervoxels)
                subset_supervoxels = pd.Index(subset_supervoxels).difference(existing_svs)

            label_df, mesh_gen = meshes_from_volume(
                brick.volume, brick.physical_box, subset_supervoxels,
                cuffs=True, capped=True,
                min_voxels=options["minimum-supervoxel-size"],
                max_voxels=options["maximum-supervoxel-size"],
                smoothing=options["mesh-parameters"]["smoothing"],
                decimation=options["mesh-parameters"]["decimation"],
                keep_normals=options["mesh-parameters"]["compute-normals"],
                progress=False
            )
            meshes = {label: mesh for label, mesh in mesh_gen}

            if (np.array(options["rescale-before-write"]) != 1.0).any():
                for m in meshes.values():
                    m.vertices_zyx[:] *= options["rescale-before-write"][::-1]  # zyx

            _write_meshes(config, resource_mgr, meshes)
            return label_df['Count'].rename('voxels').rename_axis('sv').reset_index()

        # Just one brick per partition, for better work stealing and responsive dashboard progress updates.
        target_partition_size_voxels = np.prod(self.input_service.preferred_message_shape)
        brickwall = BrickWall.from_volume_service(self.input_service, 0, None, self.client, target_partition_size_voxels, 0, self.sbm, compression='lz4_2x')
        sv_sizes = persist_and_execute(brickwall.bricks.map(process_brick))
        feather.write_feather(pd.concat(sv_sizes), 'written-sv-sizes.feather')

    def _sanitize_config(self):
        rescale_factor = self.config["gridmeshes"]["rescale-before-write"]
        if not isinstance(rescale_factor, list):
            rescale_factor = 3*[rescale_factor]
        self.config["gridmeshes"]["rescale-before-write"] = rescale_factor

    def _init_input_service(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.

        Also check the service configurations for errors.
        """
        options = self.config["gridmeshes"]
        input_config = self.config["input"]
        mgr_options = self.config["resource-manager"]

        self.mgr_client = ResourceManagerClient( mgr_options["server"], mgr_options["port"] )
        self.input_service = VolumeService.create_from_config( input_config, self.mgr_client )
        if isinstance(self.input_service.base_service, DvidVolumeService):
            assert input_config["dvid"]["supervoxels"], \
                'DVID input service config must use "supervoxels: true"'

        self.sbm = None
        if options["sparse-block-mask"]:
            with open(options["sparse-block-mask"], 'rb') as f:
                self.sbm = pickle.load(f)

        subset_supervoxels, subset_bodies = self._load_subset_labels()

        if len(subset_supervoxels) and not self.sbm:
            try:
                self.sbm = self.input_service.sparse_block_mask_for_labels(subset_supervoxels)
            except NotImplementedError:
                pass

        if len(subset_bodies) and not self.sbm:
            if not isinstance(self.input_service.base_service, DvidVolumeService):
                msg = ("Can't use 'subset-bodies' unless your input is a DVID labelmap."
                       " Try subset-supervoxels.")
                raise RuntimeError(msg)

            cfg = copy.deepcopy(input_config)
            cfg['dvid']['supervoxels'] = False
            try:
                self.sbm = VolumeService.create_from_config(cfg).sparse_block_mask_for_labels(subset_bodies)
            except NotImplementedError:
                pass

        return subset_supervoxels, subset_bodies

    def _load_subset_labels(self):
        options = self.config["gridmeshes"]
        subset_supervoxels = load_body_list(options["subset-supervoxels"], True)
        subset_bodies = load_body_list(options["subset-bodies"], True)

        if len(subset_supervoxels) and len(subset_bodies):
            raise RuntimeError("Can't use both subset-supervoxels and subset-bodies.  Choose one.")

        if len(subset_bodies) and not self.input_is_labelmap():
            raise RuntimeError("Can't use 'subset-bodies' unless your input is a DVID labelmap. Try subset-supervoxels.")

        return subset_supervoxels, subset_bodies

    def input_is_labelmap(self):
        return isinstance(self.input_service.base_service, DvidVolumeService)

    def input_is_labelmap_supervoxels(self):
        if isinstance(self.input_service.base_service, DvidVolumeService):
            return self.input_service.base_service.supervoxels
        return False

    def _prepare_output(self):
        """
        If necessary, create the output directory or
        DVID instance so that meshes can be written to it.
        """
        output_cfg = self.config["output"]
        output_fmt = self.config["gridmeshes"]["format"]

        ## directory output
        if 'directory' in output_cfg:
            # Convert to absolute so we can chdir with impunity later.
            output_cfg['directory'] = os.path.abspath(output_cfg['directory'])
            os.makedirs(output_cfg['directory'], exist_ok=True)
            return

        ##
        ## DVID output (either keyvalue or tarsupervoxels)
        ##
        (instance_type,) = output_cfg.keys()

        server = output_cfg[instance_type]['server']
        uuid = output_cfg[instance_type]['uuid']
        instance = output_cfg[instance_type]['instance']

        # If the output server or uuid is left blank,
        # we assume it should be auto-filled from the input settings.
        if server == "" or uuid == "":
            base_input = self.input_service.base_service
            if not isinstance(base_input, DvidVolumeService):
                # Can't copy from the input if the input ain't a dvid source
                raise RuntimeError("Output destination server/uuid was left blank.")

            if server == "":
                server = base_input.server
                output_cfg[instance_type]['server'] = server

            if uuid == "":
                uuid = base_input.uuid
                output_cfg[instance_type]['uuid'] = uuid

        # Resolve in case a branch was given instead of a specific uuid
        uuid = resolve_ref(server, uuid)

        if is_locked(server, uuid):
            info = fetch_server_info(server)
            if "Mode" in info and info["Mode"] == "allow writes on committed nodes":
                logger.warn(f"Output is a locked node ({uuid}), but server is in full-write mode. Proceeding.")
            elif os.environ.get("DVID_ADMIN_TOKEN", ""):
                logger.warn(f"Output is a locked node ({uuid}), but you defined DVID_ADMIN_TOKEN. Proceeding.")
            else:
                raise RuntimeError(f"Can't write to node {uuid} because it is locked.")

        if instance_type == 'tarsupervoxels' and not self.input_is_labelmap_supervoxels():
            msg = ("You shouldn't write to a tarsupervoxels instance unless "
                   "you're reading supervoxels from a labelmap input.\n"
                   "Use a labelmap input source, and set supervoxels: true")
            raise RuntimeError(msg)

        existing_instances = fetch_repo_instances(server, uuid)
        if instance in existing_instances:
            # Instance exists -- nothing to do.
            return

        if not output_cfg[instance_type]['create-if-necessary']:
            msg = (f"Output instance '{instance}' does not exist, "
                   "and your config did not specify create-if-necessary")
            raise RuntimeError(msg)

        assert instance_type in ('tarsupervoxels', 'keyvalue')

        ## keyvalue output
        if instance_type == "keyvalue":
            create_instance(server, uuid, instance, "keyvalue", tags=["type=meshes"])
            return

        ## tarsupervoxels output
        sync_instance = output_cfg["tarsupervoxels"]["sync-to"]

        if not sync_instance:
            # Auto-fill a default 'sync-to' instance using the input segmentation, if possible.
            base_input = self.input_service.base_service
            if isinstance(base_input, DvidVolumeService):
                if base_input.instance_name in existing_instances:
                    sync_instance = base_input.instance_name

        if not sync_instance:
            msg = ("Can't create a tarsupervoxels instance unless "
                   "you specify a 'sync-to' labelmap instance name.")
            raise RuntimeError(msg)

        if sync_instance not in existing_instances:
            msg = ("Can't sync to labelmap instance '{sync_instance}': "
                   "it doesn't exist on the output server.")
            raise RuntimeError(msg)

        create_tarsupervoxel_instance(server, uuid, instance, sync_instance, output_fmt)

def _determine_existing(config, all_svs):
    """
    Determine which of the given supervoxels already have
    meshes stored in the configured destination.
    """
    fmt = config["gridmeshes"]["format"]
    destination = config["output"]
    (destination_type,) = destination.keys()
    assert destination_type in ('directory', 'keyvalue', 'tarsupervoxels')

    if destination_type == 'directory':
        d = config["output"]["directory"]
        existing_svs = set()
        for sv in all_svs:
            if os.path.exists(f"{d}/{sv}.{fmt}"):
                existing_svs.add(sv)

    elif destination_type == 'tarsupervoxels':
        tsv_instance = [destination['tarsupervoxels'][k] for k in ('server', 'uuid', 'instance')]
        exists = fetch_exists(*tsv_instance, all_svs, batch_size=10_000, processes=32, show_progress=False)
        existing_svs = set(exists[exists].index)

    elif destination_type == 'keyvalue':
        logger.warning("Using skip-exists with a keyvalue output.  This might take a LONG time.")
        kv_instance = [destination['keyvalue'][k] for k in ('server', 'uuid', 'instance')]
        for sv in all_svs:
            if fetch_key(*kv_instance, f"{sv}.{fmt}", check_head=True):
                existing_svs.add(sv)

    return existing_svs

def _write_meshes(config, resource_mgr, meshes):
    options = config["gridmeshes"]
    fmt = options["format"]
    destination = config["output"]

    (destination_type,) = destination.keys()
    assert destination_type in ('directory', 'keyvalue', 'tarsupervoxels')

    binary_meshes = {f"{sv}.{fmt}": serialize_mesh(sv, mesh, None, fmt=fmt)
                        for (sv, mesh) in meshes.items()}

    if destination_type == 'directory':
        for name, mesh_bytes in binary_meshes.items():
            path = destination['directory'] + "/" + name
            with open(path, 'wb') as f:
                f.write(mesh_bytes)
    else:
        total_bytes = sum(map(len, binary_meshes.values()))
        instance = [destination[destination_type][k] for k in ('server', 'uuid', 'instance')]
        with resource_mgr.access_context(instance[0], False, 1, total_bytes):
            if destination_type == 'tarsupervoxels':
                post_load(*instance, binary_meshes)
            elif 'keyvalue' in destination:
                post_keyvalues(*instance, binary_meshes)

def serialize_mesh(sv, mesh, path=None, fmt=None):
    """
    Call mesh.serialize(), but if an error occurs,
    log it and save an .obj to 'bad-meshes'
    """
    try:
        return mesh.serialize(path, fmt)
    except:
        logger = logging.getLogger(__name__)

        msg = f"Failed to serialize mesh as {fmt}."
        if path:
            msg += f" Attempted to write to {path}"
        try:
            if not os.path.exists('bad-meshes'):
                os.makedirs('bad-meshes', exist_ok=True)
            output_path = f'bad-meshes/failed-serialization-{sv}.obj'
            mesh.serialize(output_path, 'obj')
            msg += f" Wrote to {output_path}"
            logger.error(msg)
        except Exception:
            msg += "Couldn't write as OBJ, either."
            logger.error(msg)

        return b''

