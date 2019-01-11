import os
import copy
import logging

import dask.bag
import numpy as np
import pandas as pd

from requests import HTTPError

from dvid_resource_manager.client import ResourceManagerClient

from neuclease.util import Timer, read_csv_col, read_csv_header
from neuclease.dvid.tarsupervoxels import fetch_tarfile
from neuclease.bin.decimate_existing_mesh import decimate_existing_mesh

from . import Workflow

logger = logging.getLogger(__name__)

class DecimateMeshes(Workflow):
    """
    Download pre-existing meshes from a dvid tarsupervoxels instance, and decimate them.
    
    Basically a clusterized wrapper around neuclease.bin.decimate_existing_mesh
    """
    DvidTarsupervoxelsInstanceSchema = \
    {
        "description": "Parameters specify a DVID instance",
        "type": "object",
        "required": ["server", "uuid", "tarsupervoxels-instance"],
    
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
            "tarsupervoxels-instance": {
                "description": "Name of a tarsupervoxels instance",
                "type": "string"
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
            "bodies": {
                "oneOf": [
                    {
                        "description": "A list of body IDs to generate meshes for.",
                        "type": "array",
                        "default": []
                    },
                    {
                        "description": "A CSV file containing a single column of body IDs to generate meshes for.",
                        "type": "string",
                        "default": ""
                    }
                ]
            },
            "format": {
                "description": "Format to save the meshes in",
                "type": "string",
                "enum": ["obj",     # Wavefront OBJ (.obj)
                         "drc",     # Draco (compressed) (.drc)
                         "ngmesh"], # "neuroglancer mesh" format -- a custom binary format.  Note: Data is presumed to be 8nm resolution
                "default": "obj"
            },
            "decimation-fraction": {
                "description": "Mesh decimation aims to reduce the number of \n"
                               "mesh vertices in the mesh to a fraction of the original mesh. \n"
                               "To disable decimation, use 1.0.\n",
                "type": "number",
                "minimum": 0.0000001,
                "maximum": 1.0, # 1.0 == disable
                "default": 0.1                
            },
            "output-directory": {
                "description": "Location to write decimated meshes to",
                "type": "string",
                "default": "output-meshes"
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
        # Convert input/output CSV to absolute paths
        options = self.config["decimatemeshes"]
        assert options["bodies"], \
            "No input body list provided"
        
        if isinstance(options["bodies"], str):
            assert os.path.exists(options["bodies"]), \
                f'Input file does not exist: {options["bodies"]}'


    def execute(self):
        self._sanitize_config()

        input_config = self.config["input"]
        options = self.config["decimatemeshes"]
        resource_config = self.config["resource-manager"]
        
        output_dir = options["output-directory"]
        os.makedirs(output_dir, exist_ok=True)

        resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])

        if isinstance(options["bodies"], str):
            if 'body' in read_csv_header(options["bodies"]):
                bodies = pd.read_csv(options["bodies"])['body'].drop_duplicates()
            else:
                # Just read the first column, no matter what it's named
                bodies = read_csv_col(options["bodies"], 0, np.uint64).drop_duplicates()
        else:
            assert isinstance(options["bodies"], list), \
                "input bodies must be a list or a path to a CSV file"
            bodies = options["bodies"]
        
        # Choose more partitions than cores, so that early finishers have the opportunity to steal work.
        bodies_bag = dask.bag.from_sequence(bodies, npartitions=self.total_cores() * 4)
        
        def process_body(body_id):
            output_path = f'{output_dir}/{body_id}.{options["format"]}'
            
            with resource_mgr_client.access_context( input_config["server"], True, 1, 0 ):
                try:
                    tar_bytes = fetch_tarfile( input_config["server"],
                                               input_config["uuid"],
                                               input_config["tarsupervoxels-instance"],
                                               body_id )
                except HTTPError:
                    return body_id

            decimate_existing_mesh( input_config["server"],
                                    input_config["uuid"],
                                    input_config["tarsupervoxels-instance"],
                                    body_id,
                                    options["decimation-fraction"],
                                    options["format"],
                                    output_path,
                                    tar_bytes=tar_bytes )

        with Timer(f"Decimating {len(bodies)} meshes", logger):
            problem_bodies = bodies_bag.map(process_body).compute()
        
        problem_bodies = list(filter(None, problem_bodies))
        
        if problem_bodies:
            logger.error(f"Some bodies ({len(problem_bodies)}) could not be retrieved from DVID:\n{problem_bodies}")
        
        logger.info("DONE.")

