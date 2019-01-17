import os
import copy
import getpass
import logging

import numpy as np
import pandas as pd

import dask.bag as db
from dask import delayed

from distributed import as_completed

from dvid_resource_manager.client import ResourceManagerClient

from neuclease.util import Grid
from neuclease.dvid import default_node_service

from vol2mesh import Mesh, concatenate_meshes

from libdvid import DVIDNodeService

from ..util import as_completed_synchronous
from ..brick import Brick, BrickWall
from ..volumes import VolumeService, DvidVolumeService, DvidSegmentationVolumeSchema

from .util.config_helpers import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)

class StitchedMeshes(Workflow):
    OptionsSchema = {
        "type": "object",
        "description": "Settings specific to the StitchedMeshes workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "bodies": BodyListSchema,
            "concurrent-bodies": {
                "description": "How many bodies to keep in the pipeline at once.\n"
                               "Set to 0 to use the default, which is the number of cores in your cluster.",
                "type": "integer",
                "default": 0
            },
            "scale": {
                "description": "Scale at which to fetch sparsevols.",
                "type": "integer",
                "default": 0
            },
            "block-halo": {
                "description": "Meshes will be generated in blocks before stitching.\n"
                               "This setting specifies the width of the halo around each block,\n"
                               "to ensure overlapping coverage of the computed meshes.\n"
                               "A halo of 1 pixel suffices if no decimation will be applied.\n"
                               "When using smoothing and/or decimation, a halo of 2 or more is better to avoid artifacts.",
                "type": "integer",
                "default": 1
            },
            "stitch": {
                "description": "Whether or not to 'stitch' the block meshes when combining them into a single mesh.",
                "type": "boolean",
                "default": False
            },
            "smoothing-iterations": {
                "description": "How many iterations of smoothing to apply before decimation",
                "type": "integer",
                "default": 0
            },
            "decimation-fraction": {
                "description": "Mesh decimation aims to reduce the number of \n"
                               "mesh vertices in the mesh to a fraction of the original mesh. \n"
                               "To disable decimation, use 1.0.\n"
                               "Note: If a scale other than min-scale is chosen for a particular mesh (see min-scale),\n"
                               "      the decimation fraction will be auto-increased for that mesh.",
                "type": "number",
                "minimum": 0.0000001,
                "maximum": 1.0, # 1.0 == disable
                "default": 0.1                
            },
            "format": {
                "description": "Format to save the meshes in",
                "type": "string",
                "enum": ["obj",     # Wavefront OBJ (.obj)
                         "drc",     # Draco (compressed) (.drc)
                         "ngmesh"], # "neuroglancer mesh" format -- a custom binary format.  Note: Data is presumed to be 8nm resolution
                "default": "obj"
            },
            "output-directory": {
                "description": "Location to write decimated meshes to",
                "type": "string",
                "default": "meshes"
            },
            "skip-existing": {
                "description": "If true, skip any meshes that are already present in the output directory.",
                "type": "boolean",
                "default": False,
            },
            "error-mode": {
                "description": "How to handle errors.  Either warn about them and move on, or raise immediately (useful for debugging).",
                "type": "string",
                "enum": ["warn", "raise"],
                "default": "warn"
            }
            # batch size
            # normals?
            # rescale?
            # include empty?
            
        }
    }


    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": DvidSegmentationVolumeSchema,
        "stitchedmeshes": OptionsSchema
    })


    @classmethod
    def schema(cls):
        return StitchedMeshes.Schema

    def _sanitize_config(self):
        options = self.config["stitchedmeshes"]
        if options["concurrent-bodies"] == 0:
            logger.info(f"Setting concurrent-bodies to {self.total_cores()}")
            options["concurrent-bodies"] = self.total_cores()

    def _init_service(self):
        self._sanitize_config()
        
        options = self.config["stitchedmeshes"]
        input_config = self.config["input"]
        mgr_options = self.config["resource-manager"]

        self.mgr_client = ResourceManagerClient( mgr_options["server"], mgr_options["port"] )
        self.input_service = VolumeService.create_from_config( input_config, os.getcwd(), self.mgr_client )
        assert isinstance(self.input_service, DvidVolumeService), \
            "Input must be plain dvid source, not scaled, transposed, etc."
        
        max_scale = max(self.input_service.available_scales)
        assert options["scale"] <= max_scale, \
            f"Largest available scale in the input ({max_scale}) is smaller than the scale you provided ({options['scale']})."
        
    def execute(self):
        self._init_service()
        mgr_client = self.mgr_client

        dvid_config = self.config["input"]["dvid"]
        options = self.config["stitchedmeshes"]

        is_supervoxels = self.input_service.base_service.supervoxels
        bodies = load_body_list(options["bodies"], is_supervoxels)
        
        logger.info(f"Input is {len(bodies)} bodies")
        os.makedirs(options["output-directory"], exist_ok=True)
        
        server, uuid, instance = dvid_config["server"], dvid_config["uuid"], dvid_config["segmentation-name"]
        
        def make_bricks(coord_and_block):
            coord_zyx, block_vol = coord_and_block
            logical_box = np.array((coord_zyx, coord_zyx + block_vol.shape))
            return Brick(logical_box, logical_box, block_vol, location_id=(logical_box // 64))
        
        def create_brick_mesh(brick):
            return Mesh.from_binary_vol(brick.volume, brick.physical_box)

        def create_combined_mesh(meshes):
            mesh = concatenate_meshes(meshes, False)
            if options["stitch"]:
                mesh.stitch_adjacent_faces(drop_unused_vertices=True, drop_duplicate_faces=True)
            mesh.laplacian_smooth(options["smoothing-iterations"])
            mesh.simplify(options["decimation-fraction"], in_memory=True)
            return mesh

        in_flight = 0

        # Support synchronous testing with a fake 'as_completed' object
        if hasattr(self.client, 'DEBUG'):
            result_futures = as_completed_synchronous()
        else:
            result_futures = as_completed()

        def pop_result():
            nonlocal in_flight
            r = next(result_futures)
            in_flight -= 1

            try:
                return r.result()
            except Exception as ex:
                if options["error-mode"] == "raise":
                    raise
                body = int(r.key)
                return (body, 0, 'error', str(ex))

        USER = getpass.getuser()
        results = []
        try:
            for i, body in enumerate(bodies):
                logger.info(f"Mesh #{i}: Body {body}: Starting")
                def fetch_sparsevol():
                    with mgr_client.access_context(server, True, 1, 0):
                        ns = default_node_service(server, uuid, 'flyemflows-stitchedmeshes', USER)
                        coords_zyx, blocks = ns.get_sparselabelmask(body, instance, options["scale"], is_supervoxels)
                        return list(coords_zyx.copy()), list(blocks.copy())

                # This leaves all blocks and bricks in a single partition,
                # but we're about to do a shuffle anyway when the bricks are realigned.
                coords, blocks = delayed(fetch_sparsevol, nout=2)()
                coords, blocks = db.from_delayed(coords), db.from_delayed(blocks)
                bricks = db.zip(coords, blocks).map(make_bricks)
                
                mesh_grid = Grid((64,64,64), halo=options["block-halo"])
                wall = BrickWall(None, (64,64,64), bricks)
                wall = wall.realign_to_new_grid(mesh_grid)
                
                brick_meshes = wall.bricks.map(create_brick_mesh)
                consolidated_brick_meshes = brick_meshes.repartition(1)
                combined_mesh = delayed(create_combined_mesh)(consolidated_brick_meshes)
    
                def write_mesh(mesh):
                    output_dir = options["output-directory"]
                    fmt = options["format"]
                    output_path = f'{output_dir}/{body}.{fmt}'
                    mesh.serialize(output_path)
                    return (body, len(mesh.vertices_zyx), 'success', '')
        
                # We hide the body ID in the task name, so that we can record it in pop_result
                task = delayed(write_mesh)(combined_mesh, dask_key_name=f'{body}')
                result_futures.add( self.client.compute(task) )
                in_flight += 1
                
                assert in_flight <= options["concurrent-bodies"]
                while in_flight == options["concurrent-bodies"]:
                    body, vertices, result, msg = pop_result()
                    if result == "error":
                        logger.warning(f"Body {body}: Failed to generate mesh: {msg}")
                    results.append( (body, vertices, result, msg) )
    
            # Flush the last batch of tasks
            while in_flight > 0:
                body, vertices, result, msg = pop_result()
                if result == "error":
                    logger.warning(f"Body {body}: Failed to generate mesh: {msg}")
                results.append( (body, vertices, result, msg) )
        finally:
            stats_df = pd.DataFrame(results, columns=['body', 'vertices', 'result', 'msg'])
            stats_df.to_csv('mesh-stats.csv', index=False, header=True)
            
            failed_df = stats_df.query("result != 'success'")
            if len(failed_df) > 0:
                logger.warning(f"Failed to create meshes for {len(failed_df)} bodies.  See mesh-stats.csv")


