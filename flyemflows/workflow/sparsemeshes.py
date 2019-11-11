import os
import copy
import logging
from math import log2, pow, ceil

import numpy as np
import pandas as pd

import dask.bag as db

from dvid_resource_manager.client import ResourceManagerClient

from neuclease.util import Timer
from neuclease.dvid import fetch_sparsevol_coarse, fetch_sparsevol

from vol2mesh import Mesh

from ..volumes import VolumeService, DvidVolumeService, DvidSegmentationVolumeSchema
from .util import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)

class SparseMeshes(Workflow):
    """
    This workflow 'naively' computes meshes from downloaded sparsevols.
    It will download each sparsevol at the best scale it can, ensuring that
    the bounding-box of the body at that scale doesn't exceed a certain size.
    Then it computes the entire mesh all at once (not in blocks, no stitching required).
    It saves the resulting mesh files to a directory.
    """
    
    OptionsSchema = {
        "type": "object",
        "description": "Settings specific to the SparseMeshes workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "min-scale": {
                "description": "Minimum scale at which to fetch sparsevols.\n"
                               "For individual bodies, the scale may be forced higher\n"
                               "if needed according to max-analysis-volume.",
                "type": "integer",
                "default": 0
            },
            "max-analysis-volume": {
                "description": "The above scale will be overridden (to something higher, i.e. lower resolution) \n"
                               "if the body would still be too large to generate a mesh for, as defined by this setting.\n",
                "type": "number",
                "default": 1e9 # 1 GB max
            },
            "bodies": BodyListSchema,
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
            }
        }
    }


    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": DvidSegmentationVolumeSchema,
        "sparsemeshes": OptionsSchema
    })


    @classmethod
    def schema(cls):
        return SparseMeshes.Schema

    def _init_service(self):
        options = self.config["sparsemeshes"]
        input_config = self.config["input"]
        mgr_options = self.config["resource-manager"]

        self.mgr_client = ResourceManagerClient( mgr_options["server"], mgr_options["port"] )
        self.input_service = VolumeService.create_from_config( input_config, self.mgr_client )
        assert isinstance(self.input_service, DvidVolumeService), \
            "Input must be plain dvid source, not scaled, transposed, etc."
        
        min_scale = options["min-scale"]
        max_scale = max(self.input_service.available_scales)
        assert min_scale <= max_scale, \
            f"Largest available scale in the input ({max_scale}) is smaller than the min-scale you provided ({min_scale})."
        
    def execute(self):
        self._init_service()
        mgr_client = self.mgr_client

        options = self.config["sparsemeshes"]
        max_box_voxels = options["max-analysis-volume"]
        min_scale = options["min-scale"]
        max_scale = max(self.input_service.available_scales)
        smoothing_iterations = options["smoothing-iterations"]
        decimation_fraction = options["decimation-fraction"]

        server, uuid, instance = self.input_service.base_service.instance_triple
        is_supervoxels = self.input_service.base_service.supervoxels

        bodies = load_body_list(options["bodies"], is_supervoxels)
        logger.info(f"Input is {len(bodies)} bodies")

        os.makedirs(options["output-directory"], exist_ok=True)
        
        def compute_mesh_and_write(body):
            with Timer() as timer:
                # Fetch the sparsevol to determine the bounding-box size (in scale-0 voxels)
                try:
                    with mgr_client.access_context(server, True, 1, 0):
                        # sparsevol-coarse is at scale-6
                        coords_s6 = fetch_sparsevol_coarse(server, uuid, instance, body, is_supervoxels)
                except:
                    return (body, 0, 0, 0, 0.0, timer.seconds, 'error-sparsevol-coarse')
                
                box_s6 = np.array([coords_s6.min(axis=0), 1+coords_s6.max(axis=0)])
                box_s0 = (2**6) * box_s6
                shape_s0 = (box_s0[1] - box_s0[0])
                box_voxels_s0 = np.prod(shape_s0.astype(float))
    
                # Determine the scale we'll use.
                # Solve for 'scale' in the following relationship:
                #
                #   box_voxels_s0/((2^scale)^3) <= max_box_voxels
                #
                scale = log2(pow(box_voxels_s0 / max_box_voxels, 1/3))
                scale = max(ceil(scale), min_scale)
                
                if scale > max_scale:
                    raise RuntimeError(f"Can't compute mesh for body {body}. Bounding box is {box_s0[:, ::-1].tolist()}, "
                                       f"which is too large to fit in desired RAM, even at scale {max_scale}")
    
                try:
                    with mgr_client.access_context(server, True, 1, 0):
                        coords = fetch_sparsevol(server, uuid, instance, body, is_supervoxels, scale, dtype=np.int16)
                except:
                    return (body, 0, 0, 0, 0.0, timer.seconds, 'error-sparsevol')
    
                box = box_s0 // (2**scale)
                coords -= box[0]
                num_voxels = len(coords)
                
                shape = box[1] - box[0]
                vol = np.zeros(shape, np.uint8)
                vol[(*coords.transpose(),)] = 1
                del coords
    
                try:
                    mesh = Mesh.from_binary_vol(vol, box_s0)
                except:
                    return (body, scale, num_voxels, 0, 0.0, timer.seconds, 'error-construction')
                
                del vol
                try:
                    mesh.laplacian_smooth(smoothing_iterations)
                except:
                    return (body, scale, num_voxels, 0.0, len(mesh.vertices_zyx), timer.seconds, 'error-smoothing')
                
                fraction = decimation_fraction
                if scale > min_scale:
                    # Since we're starting from a lower resolution than the user requested,
                    # Reduce the decimation we're applying accordingly.
                    # Since meshes are 2D surfaces, we approximate the difference in
                    # vertexes as the SQUARE of the difference in resolution.
                    fraction *= (2**(scale - min_scale))**2
                    fraction = min(fraction, 1.0)
    
                try:
                    mesh.simplify(fraction, in_memory=True)
                except:
                    return (body, scale, num_voxels, 0.0, len(mesh.vertices_zyx), timer.seconds, 'error-decimation')
                
                output_path = f'{options["output-directory"]}/{body}.{options["format"]}'
                mesh.serialize(output_path)
                
                return (body, scale, num_voxels, fraction, len(mesh.vertices_zyx), timer.seconds, 'success')
        
        # Run the computation -- scatter first to ensure fair distribution (fixme: does this make a difference?)
        # And use a lot of partitions to enable work-stealing if some meshes are slow to compute.
        bodies_bag = db.from_sequence(bodies, npartitions=2000)
        bodies_bag = self.client.scatter(bodies_bag).result()
        stats = bodies_bag.map(compute_mesh_and_write).compute()
        
        # Save stats
        stats_df = pd.DataFrame(stats, columns=['body', 'scale', 'voxels', 'decimation_fraction', 'vertices', 'total_seconds', 'result'])
        stats_df.to_csv('mesh-stats.csv', index=False, header=True)
        
        failed_df = stats_df.query('result != "success"')
        if len(failed_df) > 0:
            logger.warning(f"{len(failed_df)} meshes could not be generated. See mesh-stats.csv")
            logger.warning(f"Results:\n{stats_df['result'].value_counts()}")

        scales_histogram = stats_df.query("result == 'success'")['scale'].value_counts().sort_index()
        logger.info(f"Scales chosen:\n{scales_histogram}")
        

