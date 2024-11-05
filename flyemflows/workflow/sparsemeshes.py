import os
import copy
import logging
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import distributed

from dvid_resource_manager.client import ResourceManagerClient
from confiddler import flow_style
from neuclease.util import Timer, tqdm_proxy, iter_batches
from neuclease.dvid import fetch_sparsevol, set_default_dvid_session_timeout, create_tar_from_dict, post_load, post_keyvalues
from neuclease.dvid.rle import blockwise_masks_from_ranges

from vol2mesh import Mesh

from ..volumes import VolumeService, DvidVolumeService, DvidSegmentationVolumeSchema
from ..util import as_completed_synchronous
from .util import BodyListSchema, load_body_list
from .util.mesh_workflow_utils import MeshOutputSchema, prepare_mesh_output
from . import Workflow

logger = logging.getLogger(__name__)

MeshResult = namedtuple('MeshResult', 'body status buf buf_size vertex_count download_seconds meshing_seconds download_error meshing_error')


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
            "batch-size": {
                "description":
                    "How to batch body IDs into tasks in the dask scheduler.\n"
                    "For a few large bodies, 1 is good.  For many tiny bodies, larger batches\n"
                    "are ideal since otherwise the dask scheduler overhead becomes a bottleneck.\n",
                "type": "integer",
                "default": 1,
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": DvidSegmentationVolumeSchema,
        "output": MeshOutputSchema,
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

        output_cfg = self.config['output']
        options = self.config["sparsemeshes"]
        scale = options["scale"]
        halo = 1
        smoothing_iterations = options["smoothing-iterations"]
        decimation_fraction = options["decimation-fraction"]
        block_shape = options["block-shape"][::-1]
        rescale = options["rescale-factor"]
        fmt = options["format"]

        prepare_mesh_output(
            output_cfg,
            fmt,
            input_service
        )

        server, uuid, instance = input_service.base_service.instance_triple
        is_supervoxels = input_service.base_service.supervoxels

        def fetch_sparsevol_batch(bodies):
            sparsevol_ranges = {}
            with mgr_client.access_context(server, True, 1, 0):
                for body in bodies:
                    try:
                        with Timer() as timer:
                            sparsevol_ranges[body] = (
                                fetch_sparsevol(server, uuid, instance, body, scale, format='ranges'),
                                timer.seconds,
                                ''
                            )
                    except Exception as ex:
                        sparsevol_ranges[body] = (None, timer.seconds, str(ex))
            return sparsevol_ranges

        def generate_mesh_batch(sparsevol_ranges):
            set_default_dvid_session_timeout(600.0, 600.0)
            mesh_results = {}
            for body, (ranges, download_seconds, download_error) in sparsevol_ranges.items():
                with Timer() as timer:
                    if ranges is None:
                        mesh_results[body] = MeshResult(
                            body, 'failed-download', None, 0, 0, download_seconds, 0, download_error, ''
                        )
                        continue
                    try:
                        boxes, masks = blockwise_masks_from_ranges(ranges, block_shape, halo)
                        m = Mesh.from_binary_blocks(masks, boxes * 2**scale)
                        m.laplacian_smooth(smoothing_iterations)
                        m.simplify(decimation_fraction)
                        m.vertices_zyx *= rescale
                        buf = m.serialize(fmt=fmt)
                        mesh_results[body] = MeshResult(
                            body, 'success', buf, len(buf), len(m.vertices_zyx), download_seconds, timer.seconds, download_error, ''
                        )
                    except Exception as ex:
                        mesh_results[body] = MeshResult(
                            body, 'failed-meshing', 0, 0, 0, download_seconds, timer.seconds, download_error, str(ex)
                        )
            return mesh_results

        def write_meshes(batch_id, mesh_results):
            meshes_df = pd.DataFrame(mesh_results.values(), columns=MeshResult._fields)

            (destination_type,) = output_cfg.keys()
            assert destination_type in ('directory', 'directory-of-tarfiles', 'keyvalue', 'tarsupervoxels')

            meshes_df['name'] = [f"{body}.{fmt}" for body in meshes_df['body']]
            keyvalues = meshes_df.set_index('name')['buf'].to_dict()
            keyvalues = {k:v for (k,v) in keyvalues.items() if v}

            if destination_type == 'directory':
                for name, mesh_bytes in keyvalues.items():
                    path = output_cfg['directory'] + "/" + name
                    with open(path, 'wb') as f:
                        f.write(mesh_bytes)

            elif destination_type == 'directory-of-tarfiles':
                batch_name = f"batch-{batch_id}"
                batch_dir = f"{output_cfg['directory-of-tarfiles']}/{batch_id // 1000}"
                os.makedirs(batch_dir, exist_ok=True)
                tar_path = Path(f"{batch_dir}/{batch_name}.tar")
                create_tar_from_dict(keyvalues, tar_path)

            else:
                # Set the timeouts here, inside the worker process.
                # (Is this necessary? Probably doesn't hurt.)
                set_default_dvid_session_timeout(
                    output_cfg[destination_type]["timeout"],
                    output_cfg[destination_type]["timeout"]
                )
                instance = [output_cfg[destination_type][k] for k in ('server', 'uuid', 'instance')]

                total_bytes = meshes_df['buf_size'].sum()
                with mgr_client.access_context(instance[0], False, 1, total_bytes):
                    if destination_type == 'tarsupervoxels':
                        post_load(*instance, keyvalues)
                    elif destination_type == 'keyvalue':
                        post_keyvalues(*instance, keyvalues)

            return meshes_df.drop(columns=['buf'])

        def process_batch(bodies, batch_id):
            spv_ranges = fetch_sparsevol_batch(bodies)
            mesh_results = generate_mesh_batch(spv_ranges)
            results_df = write_meshes(batch_id, mesh_results)
            results_df['batch_id'] = batch_id
            return batch_id, results_df

        bodies = load_body_list(options["bodies"], is_supervoxels)
        body_batches = iter_batches(bodies, options["batch-size"])
        batch_ids = np.arange(len(body_batches))
        logger.info(f"Input is {len(bodies)} bodies ({len(body_batches)} batches)")

        futures = self.client.map(process_batch, body_batches, batch_ids)

        # Support synchronous testing with a fake 'as_completed' object
        if hasattr(self.client, 'DEBUG'):
            ac = as_completed_synchronous(futures, with_results=True)
        else:
            ac = distributed.as_completed(futures, with_results=True)

        all_results_dfs = []
        try:
            for _fut, result in tqdm_proxy(ac, total=len(futures)):
                batch_id, results_df = result
                all_results_dfs.append(results_df)

                if results_df['download_error'].any():
                    b = results_df.loc[results_df['download_error'].astype(bool), 'body'].tolist()
                    logger.warning(f"Batch {batch_id}: Failed to download bodies: {b}")

                if results_df['meshing_error'].any():
                    b = results_df.loc[results_df['meshing_error'].astype(bool), 'body'].tolist()
                    logger.warning(f"Batch {batch_id}: Failed to generate meshes for bodies: {b}")
        finally:
            if all_results_dfs:
                all_results_df = pd.concat(all_results_dfs)
                feather.write_feather(all_results_df, 'mesh-stats.feather')

                failed_df = all_results_df.query('status != "success"')
                if len(failed_df) > 0:
                    logger.warning(f"{len(failed_df)} meshes could not be generated. See mesh-stats.feather")
                    logger.warning("Result summary:\n")
                    logger.warning(f"{all_results_df['result'].value_counts()}")
