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
    
    TODO: Save mesh stats to a csv file.
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
        
        skip_existing = options['skip-existing']
        def process_body(body_id):
            output_path = f'{output_dir}/{body_id}.{options["format"]}'
            if skip_existing and os.path.exists(output_path):
                return (body_id, 0, 'skipped')
            
            with resource_mgr_client.access_context( input_config["server"], True, 1, 0 ):
                try:
                    tar_bytes = fetch_tarfile( input_config["server"],
                                               input_config["uuid"],
                                               input_config["tarsupervoxels-instance"],
                                               body_id )
                except HTTPError:
                    # FIXME: Better to log the exception strings to a file
                    return (body_id, 0, 'error-fetch')

            try:
                vertex_count = decimate_existing_mesh( input_config["server"],
                                                       input_config["uuid"],
                                                       input_config["tarsupervoxels-instance"],
                                                       body_id,
                                                       options["decimation-fraction"],
                                                       options["format"],
                                                       output_path,
                                                       tar_bytes=tar_bytes )
            except:
                return (body_id, 0, 'error-generate')

            return (body_id, vertex_count, 'success')

        with Timer(f"Decimating {len(bodies)} meshes", logger):
            stats = bodies_bag.map(process_body).compute()
        
        
        stats_df = pd.DataFrame(stats, columns=['body', 'vertices', 'result'])
        stats_df.to_csv('mesh-stats.csv', index=False, header=True)

        failed_df = stats_df.query('result != "success"')
        if len(failed_df) > 0:
            logger.warning(f"{len(failed_df)} meshes could not be generated. See mesh-stats.csv")
            logger.warning(f"Results:\n{stats_df['result'].value_counts()}")
