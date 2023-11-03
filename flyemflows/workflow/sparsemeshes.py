import os
import copy
import logging

import pandas as pd
import distributed

from dvid_resource_manager.client import ResourceManagerClient
from confiddler import flow_style
from neuclease.util import Timer, tqdm_proxy
from neuclease.dvid import fetch_sparsevol, set_default_dvid_session_timeout
from neuclease.dvid.rle import blockwise_masks_from_ranges

from vol2mesh import Mesh

from ..volumes import VolumeService, DvidVolumeService, DvidSegmentationVolumeSchema
from ..util import as_completed_synchronous
from .util import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)


class SparseMeshes(Workflow):
    """
    Compute meshes for a set of bodies from their sparsevol representations.
    It saves the resulting mesh files to a directory.
    """

    OptionsSchema = {
        "type": "object",
        "description": "Settings specific to the SparseMeshes workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "bodies": BodyListSchema,
            "scale": {
                "description":
                    "Scale at which to fetch sparsevols.\n"
                    "Mesh vertices will be converted to scale-0.\n",
                "type": "integer",
                "default": 0
            },
            "rescale-factor": {
                "description": "Optionally rescale the vertex positions before storing the mesh.\n",
                "type": "array",
                "items": { "type": "number" },
                "minItems": 3,
                "maxItems": 3,
                "default": flow_style([1, 1, 1])
            },
            "block-shape": {
                "description": "The mesh will be generated in blocks and the blocks will be stitched together.\n",
                "type": "array",
                "items": { "type": "integer" },
                "minItems": 3,
                "maxItems": 3,
                "default": flow_style([-1,-1,-1])
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
                "maximum": 1.0,  # 1.0 == disable
                "default": 0.1
            },
            "format": {
                "description": "Format to save the meshes in",
                "type": "string",
                "enum": [
                    "obj",      # Wavefront OBJ (.obj)
                    "drc",      # Draco (compressed) (.drc)
                    "ngmesh"    # "neuroglancer mesh" format -- a custom binary format.
                                # Note: Data is presumed to be 8nm resolution, so you need to use rescale-facctor
                ],
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

    def execute(self):
        input_config = self.config["input"]
        mgr_options = self.config["resource-manager"]
        mgr_client = ResourceManagerClient(mgr_options["server"], mgr_options["port"])
        input_service = VolumeService.create_from_config(input_config, mgr_client)
        assert isinstance(input_service, DvidVolumeService), \
            "Input must be plain dvid source, not scaled, transposed, etc."

        options = self.config["sparsemeshes"]
        scale = options["scale"]
        halo = 1
        smoothing_iterations = options["smoothing-iterations"]
        decimation_fraction = options["decimation-fraction"]
        block_shape = options["block-shape"][::-1]
        output_dir = options["output-directory"]
        rescale = options["rescale-factor"]
        fmt = options["format"]

        server, uuid, instance = input_service.base_service.instance_triple
        is_supervoxels = input_service.base_service.supervoxels

        bodies = load_body_list(options["bodies"], is_supervoxels)
        logger.info(f"Input is {len(bodies)} bodies")

        os.makedirs(output_dir, exist_ok=True)

        def compute_mesh_and_write(body):
            set_default_dvid_session_timeout(600.0, 600.0)
            with Timer() as timer:
                try:
                    with mgr_client.access_context(server, True, 1, 0):
                        rng = fetch_sparsevol(server, uuid, instance, body, scale, format='ranges')

                    boxes, masks = blockwise_masks_from_ranges(rng, block_shape, halo)
                    m = Mesh.from_binary_blocks(masks, boxes * 2**scale)
                    m.laplacian_smooth(smoothing_iterations)
                    m.simplify(decimation_fraction)
                    m.vertices_zyx *= rescale
                    output_path = f'{output_dir}/{body}.{fmt}'
                    m.serialize(output_path)
                    return (body, len(m.vertices_zyx), timer.seconds, 'success', '')
                except Exception as ex:
                    return (body, 0, timer.seconds, 'failed', str(ex))

        futures = self.client.map(compute_mesh_and_write, bodies)

        # Support synchronous testing with a fake 'as_completed' object
        if hasattr(self.client, 'DEBUG'):
            ac = as_completed_synchronous(futures, with_results=True)
        else:
            ac = distributed.as_completed(futures, with_results=True)

        try:
            stats = []
            for f, r in tqdm_proxy(ac, total=len(futures)):
                stats.append(r)
                body, vertices, total_seconds, result, err = r
                if result != "success":
                    logger.warning(f"Body {body} failed: {err}")
        finally:
            stats_df = pd.DataFrame(stats, columns=['body', 'vertices', 'total_seconds', 'result', 'errors'])
            stats_df.to_csv('mesh-stats.csv', index=False, header=True)

        failed_df = stats_df.query('result != "success"')
        if len(failed_df) > 0:
            logger.warning(f"{len(failed_df)} meshes could not be generated. See mesh-stats.csv")
            logger.warning(f"Results:\n{stats_df['result'].value_counts()}")
