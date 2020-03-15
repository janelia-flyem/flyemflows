import os
import copy
import logging

import dask.bag
import numpy as np
import pandas as pd

from requests import HTTPError

from confiddler import flow_style
from dvid_resource_manager.client import ResourceManagerClient

from neuclease.util import Timer
from neuclease.dvid import resolve_ref, fetch_instance_info, fetch_tarfile, fetch_mutation_id
from neuclease.bin.decimate_existing_mesh import decimate_existing_mesh

from .util import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)

class DecimateMeshes(Workflow):
    """
    Download pre-existing meshes from a dvid tarsupervoxels instance, and decimate them.
    
    Basically a clusterized wrapper around neuclease.bin.decimate_existing_mesh
    
    TODO: Save mesh stats to a csv file.
    """
    DvidTarsupervoxelsInstanceSchema = \
    {
        "description": "Parameters specify a DVID instance",
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
    
    DecimateMeshesOptionsSchema = \
    {
        "type": "object",
        "description": "Settings specific to the DecimateMeshes workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "bodies": BodyListSchema,
            "skip-existing": {
                "description": "If true, skip any meshes that are already present in the output directory.",
                "type": "boolean",
                "default": False,
            },
            "format": {
                "description": "Format to save the meshes in",
                "type": "string",
                "enum": ["obj",     # Wavefront OBJ (.obj)
                         "drc",     # Draco (compressed) (.drc)
                         "ngmesh"], # "neuroglancer mesh" format -- a custom binary format.  Note: Data is presumed to be 8nm resolution
                "default": "obj"
            },
            "decimation": {
                "description": "Mesh decimation aims to reduce the number of \n"
                               "mesh vertices in the mesh to a fraction of the original mesh. \n"
                               "To disable decimation, use 1.0.\n",
                "type": "number",
                "minimum": 0.0000001,
                "maximum": 1.0, # 1.0 == disable
                "default": 0.1                
            },
            "max-vertices": {
                "description": "Ensure that meshes have no more vertices than specified by this setting.\n"
                               "That is, decrease the decimation fraction if necessary bring the mesh vertex count below this number.\n",
                "type": "number",
                "default": 1e9 # very large
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
            "output-directory": {
                "description": "Location to write decimated meshes to",
                "type": "string",
                "default": "meshes"
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": DvidTarsupervoxelsInstanceSchema,
        "decimatemeshes": DecimateMeshesOptionsSchema
    })


    @classmethod
    def schema(cls):
        return DecimateMeshes.Schema

    def _sanitize_config(self):
        """
        - Normalize/overwrite certain config values
        - Check for config mistakes
        - Simple sanity checks
        """
        # Resolve uuid if necessary (e.g. 'master' -> abc123)        
        dvid_cfg = self.config["input"]["dvid"]
        dvid_cfg["uuid"] = resolve_ref(dvid_cfg["server"], dvid_cfg["uuid"])
        
        # Convert input/output CSV to absolute paths
        options = self.config["decimatemeshes"]
        assert options["bodies"], \
            "No input body list provided"
        
        if isinstance(options["bodies"], str):
            assert os.path.exists(options["bodies"]), \
                f'Input file does not exist: {options["bodies"]}'


    def execute(self):
        self._sanitize_config()

        input_config = self.config["input"]["dvid"]
        options = self.config["decimatemeshes"]
        resource_config = self.config["resource-manager"]
        
        output_dir = options["output-directory"]
        os.makedirs(output_dir, exist_ok=True)

        resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])

        bodies = load_body_list(options["bodies"], False)

        # Choose more partitions than cores, so that early finishers have the opportunity to steal work.
        bodies_bag = dask.bag.from_sequence(bodies, npartitions=self.total_cores() * 10)
        
        server = input_config["server"]
        uuid = input_config["uuid"]
        tsv_instance = input_config["tarsupervoxels-instance"]

        # Determine segmentation instance
        info = fetch_instance_info(server, uuid, tsv_instance)
        seg_instance = info["Base"]["Syncs"][0]
        input_format = info["Extended"]["Extension"]
        
        skip_existing = options['skip-existing']
        
        if np.array(options["rescale"] == 1.0).all() and options["format"] == "ngmesh" and input_format != "ngmesh":
            logger.warning("*** You are converting to ngmesh format, but you have not specified a rescale parameter! ***")

        def process_body(body_id):
            output_path = f'{output_dir}/{body_id}.{options["format"]}'
            if skip_existing and os.path.exists(output_path):
                return (body_id, 0, 0.0, 0, 'skipped', 0)
            
            with resource_mgr_client.access_context( input_config["server"], True, 1, 0 ):
                try:
                    mutid = fetch_mutation_id(server, uuid, seg_instance, body_id)
                except HTTPError:
                    # FIXME: Better to log the exception strings to a file
                    return (body_id, 0, 0.0, 0, 'error-mutid', 0)

                try:
                    tar_bytes = fetch_tarfile(server, uuid, tsv_instance, body_id)
                except HTTPError:
                    # FIXME: Better to log the exception strings to a file
                    return (body_id, 0, 0.0, 0, 'error-fetch', mutid)

            try:
                vertex_count, fraction, orig_vertices = \
                    decimate_existing_mesh( server, uuid, tsv_instance, body_id,
                                            options["decimation"], options["max-vertices"], options["rescale"], options["format"],
                                            output_path,
                                            tar_bytes=tar_bytes )
            except:
                return (body_id, 0, 0.0, 0, 'error-generate', mutid)

            return (body_id, vertex_count, fraction, orig_vertices, 'success', mutid)

        with Timer(f"Decimating {len(bodies)} meshes", logger):
            stats = bodies_bag.map(process_body).compute()
        
        
        stats_df = pd.DataFrame(stats, columns=['body', 'vertices', 'decimation', 'orig_vertices', 'result', 'mutid'])
        stats_df['uuid'] = uuid

        stats_df.to_csv('mesh-stats.csv', index=False, header=True)
        np.save('mesh-stats.npy', stats_df.to_records(index=False))

        failed_df = stats_df.query('result != "success"')
        if len(failed_df) > 0:
            logger.warning(f"{len(failed_df)} meshes could not be generated. See mesh-stats.csv")
            logger.warning(f"Results:\n{stats_df['result'].value_counts()}")
