import os
import pickle
import copy
import logging
from itertools import starmap
from functools import partial

import distributed
import dask.config
import numpy as np
import pandas as pd

from confiddler import flow_style
from dvid_resource_manager.client import ResourceManagerClient

from neuclease.util import Timer, compute_parallel, tqdm_proxy
from neuclease.dvid import (resolve_ref, fetch_instance_info, fetch_tarfile,
                            create_instance, create_tarsupervoxel_instance, fetch_repo_instances,
                            is_locked, fetch_server_info, post_load, post_keyvalues)

from vol2mesh.mesh import Mesh

from ..util import as_completed_synchronous
from .util import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)


class SVDecimate(Workflow):
    """
    Download pre-existing supervoxel meshes from
    a dvid tarsupervoxels instance and decimate them.

    This workflow can also be used to convert the mesh
    files from one format to another, or rescale the vertex
    coordinates.
    """
    TarsupervoxelsInputSchema = \
    {
        "description": "Parameters specify a DVID tarsupervoxels instance",
        "type": "object",

        "default": {},
        "required": ["dvid"],
        "additionalProperties": False,
        "properties": {
            "dvid": {
                "default": {},
                "type": "object",
                "required": ["server", "uuid", "tarsupervoxels-instance"],
                "additionalProperties": False,
                "properties": {
                    "server": {
                        "description": "location of DVID server to READ.",
                        "type": "string",
                    },
                    "uuid": {
                        "description": "version node for READING segmentation",
                        "type": "string"
                    },
                    "tarsupervoxels-instance": {
                        "description": "Name of a tarsupervoxels instance",
                        "type": "string"
                    }
                }
            }
        }
    }

    GenericDvidInstanceSchema = \
    {
        "description": "Parameters to specify a generic dvid instance (server/uuid/instance).\n"
                       "Omitted values will be copied from the input, or given default values.",
        "type": "object",
        "required": ["server", "uuid"],

        # Must not have default. (Appears below in a 'oneOf' context.)
        # "default": {},
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
                # Must not have default. (Appears below in a 'oneOf' context.)
                # "default": ""
            }
        }
    }

    SVDecimateOptionSchema = \
    {
        "type": "object",
        "description": "Settings specific to the SVDecimate workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "bodies": BodyListSchema,
            "decimation": {
                "description": "Mesh decimation aims to reduce the number of \n"
                               "mesh vertices in the mesh to a fraction of the original mesh. \n"
                               "To disable decimation, use 1.0.\n",
                "type": "number",
                "minimum": 0.0000001,
                "maximum": 1.0,  # 1.0 == disable
                "default": 0.1
            },
            "decimation-library": {
                "type": "string",
                "enum": ["openmesh", "fq-in-memory", "fq-via-disk"],
                "default": "openmesh"
            },
            "max-sv-vertices": {
                "description": "Ensure that meshes have no more vertices than specified by this setting.\n"
                               "That is, decrease the decimation fraction if necessary bring the mesh vertex count below this number.\n",
                "type": "number",
                "default": 1e9  # very large
            },
            "max-body-vertices": {
                "description": "If necessary, reduce the decimation fraction to ensure that the total vertex\n"
                               "count across all supervoxels in each body does not exceed this number.\n",
                "type": "number",
                "default": 1e9,  # effectively unlimited
            },
            "rescale": {
                "description": "How to multiply the mesh vertex coordinates before saving the mesh.\n"
                               "Typically very important when converting to ngmesh format from some other format.\n",
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "default": flow_style([1.0, 1.0, 1.0])
            },
            "processes-per-body": {
                "description": "Parallelism to use when processing supervoxel meshes for each body.\n"
                               "Bodies are processed in a single dask task, but further parallelism may be desirable within that task.\n",
                "type": "integer",
                "default": 1,
                "minimum": 1
            },
            "format": {
                "description": "Format in which to save the meshes",
                "type": "string",
                "enum": ["obj",      # Wavefront OBJ (.obj)
                        "drc",      # Draco (compressed) (.drc)
                        "ngmesh"],  # "neuroglancer mesh" format -- a custom binary format.  Note: Data is presumed to be 8nm resolution
                "default": "obj"
            },
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": TarsupervoxelsInputSchema,
        "output": {
            "oneOf": [
                DirectoryOutputSchema,
                TarsupervoxelsOutputSchema,
                KeyvalueOutputSchema,
            ],
            "default": {"directory": "sv-meshes"}
        },
        "svdecimate": SVDecimateOptionSchema
    })

    @classmethod
    def schema(cls):
        return SVDecimate.Schema

    def execute(self):
        self._sanitize_config()
        self._prepare_output()

        input_config = self.config["input"]["dvid"]
        output_config = self.config["output"]
        options = self.config["svdecimate"]
        resource_config = self.config["resource-manager"]

        resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])

        server = input_config["server"]
        uuid = input_config["uuid"]
        tsv_instance = input_config["tarsupervoxels-instance"]

        bodies = load_body_list(options["bodies"], False)

        # Determine segmentation instance
        info = fetch_instance_info(server, uuid, tsv_instance)
        input_format = info["Extended"]["Extension"]

        output_format = options["format"]

        if np.array(options["rescale"] == 1.0).all() and output_format == "ngmesh" and input_format != "ngmesh":
            logger.warning("*** You are converting to ngmesh format, but you have not specified a rescale parameter! ***")

        decimation_lib = options["decimation-library"]
        max_sv_vertices = options["max-sv-vertices"]
        max_body_vertices = options["max-body-vertices"]
        num_procs = options["processes-per-body"]

        def process_body(body_id):
            with resource_mgr_client.access_context( input_config["server"], True, 1, 0 ):
                tar_bytes = fetch_tarfile(server, uuid, tsv_instance, body_id)

            sv_meshes = Mesh.from_tarfile(tar_bytes, concatenate=False)
            sv_meshes = {int(os.path.splitext(name)[0]): m for name, m in sv_meshes.items()}

            total_body_vertices = sum([len(m.vertices_zyx) for m in sv_meshes.values()])
            decimation = min(1.0, max_body_vertices / total_body_vertices)

            try:
                _process_sv = partial(process_sv, decimation, decimation_lib, max_sv_vertices, output_format)
                if num_procs <= 1:
                    output_table = [*starmap(_process_sv, sv_meshes.items())]
                else:
                    output_table = compute_parallel(_process_sv, sv_meshes.items(), starmap=True, processes=num_procs, ordered=False, show_progress=False)

                cols = ['sv', 'orig_vertices', 'final_vertices', 'final_decimation', 'effective_decimation', 'mesh_bytes']
                output_df = pd.DataFrame(output_table, columns=cols)
                output_df['body'] = body_id
                output_df['error'] = ""
                write_sv_meshes(output_df, output_config, output_format, resource_mgr_client)
            except Exception as ex:
                svs = [*sv_meshes.keys()]
                orig_vertices = [len(m.vertices_zyx) for m in sv_meshes.values()]
                output_df = pd.DataFrame({'sv': svs, 'orig_vertices': orig_vertices})
                output_df['final_vertices'] = -1
                output_df['final_decimation'] = -1
                output_df['effective_decimation'] = -1
                output_df['mesh_bytes'] = -1
                output_df['body'] = body_id
                output_df['error'] = str(ex)

            return output_df.drop(columns=['mesh_bytes'])

        futures = self.client.map(process_body, bodies)

        # Support synchronous testing with a fake 'as_completed' object
        if hasattr(self.client, 'DEBUG'):
            ac = as_completed_synchronous(futures, with_results=True)
        else:
            ac = distributed.as_completed(futures, with_results=True)

        try:
            stats = []
            for f, r in tqdm_proxy(ac, total=len(futures)):
                stats.append(r)
                if (r['error'] != "").any():
                    body = r['body'].iloc[0]
                    logger.warning(f"Body {body} failed!")

        finally:
            stats_df = pd.concat(stats)
            stats_df.to_csv('mesh-stats.csv', index=False, header=True)
            with open('mesh-stats.pkl', 'wb') as f:
                pickle.dump(stats_df, f)

    def _sanitize_config(self):
        # Convert input/output CSV to absolute paths
        options = self.config["svdecimate"]
        assert options["bodies"], "No input body list provided"
        if isinstance(options["bodies"], str) and options["bodies"].endswith(".csv"):
            assert os.path.exists(options["bodies"]), \
                f'Input file does not exist: {options["bodies"]}'

        is_distributed = self.config["cluster-type"] not in ("synchronous", "processes")
        needs_multiprocessing = (options["processes-per-body"] > 1)
        workers_are_daemon = dask.config.get('distributed.worker.daemon', True)
        if is_distributed and needs_multiprocessing and workers_are_daemon:
            msg = ("This workflow uses multiprocessing, so you must configure your dask workers NOT to be daemons.\n"
                   "In your dask-config, set distributed.worker.daemon: false")
            raise RuntimeError(msg)

    def _prepare_output(self):
        """
        If necessary, create the output directory or
        DVID instance so that meshes can be written to it.
        """
        input_cfg = self.config["input"]
        output_cfg = self.config["output"]
        options = self.config["svdecimate"]

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
            assert "dvid" in input_cfg
            if server == "":
                output_cfg[instance_type]['server'] = input_cfg["dvid"]["server"]

            if uuid == "":
                output_cfg[instance_type]['uuid'] = input_cfg["dvid"]["uuid"]

        # Resolve in case a branch was given instead of a specific uuid
        server = output_cfg[instance_type]['server']
        uuid = output_cfg[instance_type]['uuid']
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
            info = fetch_instance_info(*[input_cfg["dvid"][k] for k in ("server", "uuid", "tarsupervoxels-instance")])
            syncs = info['Base']['Syncs']
            if syncs:
                sync_instance = syncs[0]

        if not sync_instance:
            msg = ("Can't create a tarsupervoxels instance unless "
                   "you specify a 'sync-to' labelmap instance name.")
            raise RuntimeError(msg)

        if sync_instance not in existing_instances:
            msg = ("Can't sync to labelmap instance '{sync_instance}': "
                   "it doesn't exist on the output server.")
            raise RuntimeError(msg)

        create_tarsupervoxel_instance(server, uuid, instance, sync_instance, options["format"])


def process_sv(decimation, decimation_lib, max_sv_vertices, output_format, sv: int, mesh: Mesh):
    try:
        orig_vertices = len(mesh.vertices_zyx)
        if orig_vertices == 0:
            final_decimation = 1.0
        else:
            final_decimation = min(decimation, max_sv_vertices / len(mesh.vertices_zyx))
            if decimation_lib == "openmesh":
                mesh.simplify_openmesh(final_decimation)
            elif decimation_lib == "fq-in-memory":
                mesh.simplify(decimation, True)
            elif decimation_lib == "fq-via-disk":
                mesh.simplify(decimation, False)
            else:
                raise AssertionError()

        final_vertices = len(mesh.vertices_zyx)
        effective_decimation = final_vertices / orig_vertices
        mesh_bytes = mesh.serialize(fmt=output_format)
        return sv, orig_vertices, final_vertices, final_decimation, effective_decimation, mesh_bytes
    except Exception as ex:
        raise RuntimeError(f"Failed processing SV {sv}: {type(ex)}") from ex


def write_sv_meshes(output_df, output_cfg, output_format, resource_mgr_client):
    fmt = output_format
    (destination_type,) = output_cfg.keys()
    assert destination_type in ('directory', 'keyvalue', 'tarsupervoxels')

    if destination_type == 'directory':
        for row in output_df.itertuples():
            p = f"{output_cfg['directory']}/{row.sv}.{fmt}"
            with open(p, 'wb') as f:
                f.write(row.mesh_bytes)
        return

    location = [output_cfg[destination_type][k] for k in ('server', 'uuid', 'instance')]
    total_bytes = sum(map(len, output_df['mesh_bytes']))

    names = [f"{sv}.{fmt}" for sv in output_df['sv']]
    keyvalues = dict(zip(names, output_df['mesh_bytes']))
    with resource_mgr_client.access_context(location[0], False, 1, total_bytes):
        if 'tarsupervoxels' in output_cfg:
            post_load(*location, keyvalues)
        elif 'keyvalue' in output_cfg:
            post_keyvalues(*location, keyvalues)
