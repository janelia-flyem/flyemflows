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
from neuclease.dvid import (
    fetch_sparsevol, fetch_instance_info, fetch_lastmod,
    set_default_dvid_session_timeout, create_tar_from_dict, post_keyvalues
)
from neuclease.misc.skeletonize import skeletonize_neuron_from_ranges

from ..volumes import VolumeService, DvidVolumeService, DvidSegmentationVolumeSchema
from ..util import as_completed_synchronous
from .util import BodyListSchema, load_body_list
from .util.skeleton_workflow_utils import SkeletonOutputSchema, prepare_skeleton_output
from . import Workflow

logger = logging.getLogger(__name__)

SkeletonResult = namedtuple(
    'SkeletonResult',
    'body status buf buf_size node_count download_seconds skeletonize_seconds download_error skeletonize_error'
)


class SparseSkeletons(Workflow):
    """
    Compute skeletons for a set of bodies from their sparsevol representations,
    using neuclease's blockwise skeletonize_neuron().  Saves the resulting
    skeletons (SWC or neuroglancer format) to a directory or DVID keyvalue instance.
    """
    OptionsSchema = {
        "type": "object",
        "description": "Settings specific to the SparseSkeletons workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "bodies": BodyListSchema,
            "scale": {
                "description":
                    "Scale at which to fetch sparsevols.\n"
                    "Skeleton coordinates are converted to scale-0 voxel units regardless.\n",
                "type": "integer",
                "default": 2
            },
            "block-shape": {
                "description":
                    "The sparsevol is skeletonized in overlapping blocks of this shape (XYZ),\n"
                    "which bounds peak RAM usage.  The blockwise skeletons are stitched together.\n",
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
                "default": flow_style([128, 128, 128])
            },
            "halo": {
                "description":
                    "The overlap (in scale-adjusted voxels) between neighboring blocks.\n"
                    "A larger halo reduces skeleton artifacts near block boundaries.\n",
                "type": "integer",
                "default": 16
            },
            "closing-radius": {
                "description":
                    "Radius of the morphological closing applied to each block's mask before\n"
                    "skeletonization.  Use 0 to disable closing.\n",
                "type": "integer",
                "default": 5
            },
            "heal-max-distance": {
                "description":
                    "If greater than 0, disconnected skeleton fragments are reconnected via\n"
                    "bridging edges no longer than this distance, specified in NANOMETERS.\n"
                    "Use 0 to leave distinct connected components unconnected.\n",
                "type": "number",
                "default": 0.0
            },
            "format": {
                "description": "Format in which to save the skeletons.",
                "type": "string",
                "enum": [
                    "swc",           # SWC text
                    "neuroglancer"   # neuroglancer "precomputed" binary skeleton
                ],
                "default": "swc"
            },
            "processing-threads": {
                "description":
                    "Number of threads skeletonize_neuron() uses internally to process a\n"
                    "single body's blocks.  Leave at 1 to rely on dask for parallelism across bodies.\n",
                "type": "integer",
                "default": 1
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
        "output": SkeletonOutputSchema,
        "sparseskeletons": OptionsSchema
    })

    @classmethod
    def schema(cls):
        return SparseSkeletons.Schema

    def execute(self):
        input_config = self.config["input"]
        mgr_options = self.config["resource-manager"]
        mgr_client = ResourceManagerClient(mgr_options["server"], mgr_options["port"])
        input_service = VolumeService.create_from_config(input_config, mgr_client)
        assert isinstance(input_service, DvidVolumeService), \
            "Input must be plain dvid source, not scaled, transposed, etc."
        assert not input_service.base_service.supervoxels, \
            "SparseSkeletons operates on bodies, not supervoxels."

        output_cfg = self.config['output']
        options = self.config["sparseskeletons"]
        scale = options["scale"]
        block_shape = options["block-shape"][::-1]  # XYZ config -> ZYX
        halo = options["halo"]
        closing_radius = options["closing-radius"]
        heal_max_distance = options["heal-max-distance"] or None
        fmt = options["format"]
        threads = options["processing-threads"]

        prepare_skeleton_output(output_cfg, input_service)

        server, uuid, instance = input_service.base_service.instance_triple

        # The physical voxel size (nm, XYZ) is constant for the instance, so fetch
        # it once here (rather than inside every task).
        voxel_size_xyz = fetch_instance_info(server, uuid, instance)['Extended']['VoxelSize']

        need_mutid = (fmt == 'swc')

        def fetch_sparsevol_batch(bodies):
            # Fetch the (DVID) inputs for a batch of bodies, holding the resource
            # manager only for the duration of the fetches -- NOT for the (CPU-bound)
            # skeletonization, which happens later in generate_skeleton_batch().
            fetched = {}
            with mgr_client.access_context(server, True, 1, 0):
                for body in bodies:
                    with Timer() as timer:
                        try:
                            ranges = fetch_sparsevol(server, uuid, instance, body, scale, format='ranges')
                            mutid = None
                            if need_mutid:
                                mutid = fetch_lastmod(server, uuid, instance, body)["mutation id"]
                            fetched[body] = (ranges, mutid, timer.seconds, '')
                        except Exception as ex:
                            fetched[body] = (None, None, timer.seconds, str(ex))
            return fetched

        def generate_skeleton_batch(fetched):
            results = {}
            for body, (ranges, mutid, download_seconds, download_error) in fetched.items():
                with Timer() as timer:
                    if ranges is None:
                        results[body] = SkeletonResult(
                            body, 'failed-download', None, 0, 0, download_seconds, 0, download_error, ''
                        )
                        continue
                    try:
                        buf = skeletonize_neuron_from_ranges(
                            ranges,
                            scale=scale,
                            block_shape=block_shape,
                            halo=halo,
                            closing_radius=closing_radius,
                            heal_max_distance=heal_max_distance,
                            voxel_size_xyz=voxel_size_xyz,
                            format=fmt,
                            threads=threads,
                            uuid=uuid,
                            segmentation_instance=instance,
                            mutid=mutid,
                        )
                        buf = buf.encode('utf-8') if isinstance(buf, str) else buf
                        node_count = buf.count(b'\n') if fmt == 'swc' else 0
                        results[body] = SkeletonResult(
                            body, 'success', buf, len(buf), node_count, download_seconds, timer.seconds, download_error, ''
                        )
                    except Exception as ex:
                        results[body] = SkeletonResult(
                            body, 'failed-skeletonize', None, 0, 0, download_seconds, timer.seconds, download_error, str(ex)
                        )
            return results

        def _skeleton_name(body, for_directory):
            # SWC skeletons are conventionally named '{body}_swc' in DVID keyvalue
            # instances, but '{body}.swc' as files.  Neuroglancer skeletons are
            # keyed/named by the bare body id.
            if fmt != 'swc':
                return str(body)
            return f"{body}.swc" if for_directory else f"{body}_swc"

        def write_skeletons(batch_id, results):
            results_df = pd.DataFrame(results.values(), columns=SkeletonResult._fields)

            (destination_type,) = output_cfg.keys()
            assert destination_type in ('directory', 'directory-of-tarfiles', 'keyvalue')

            success_df = results_df.query('status == "success"')

            if destination_type == 'directory':
                for row in success_df.itertuples(index=False):
                    path = output_cfg['directory'] + "/" + _skeleton_name(row.body, for_directory=True)
                    with open(path, 'wb') as f:
                        f.write(row.buf)

            elif destination_type == 'directory-of-tarfiles':
                keyvalues = {
                    _skeleton_name(row.body, for_directory=False): row.buf
                    for row in success_df.itertuples(index=False)
                }
                batch_name = f"batch-{batch_id}"
                batch_dir = f"{output_cfg['directory-of-tarfiles']}/{batch_id // 1000}"
                os.makedirs(batch_dir, exist_ok=True)
                tar_path = Path(f"{batch_dir}/{batch_name}.tar")
                create_tar_from_dict(keyvalues, tar_path)

            else:  # keyvalue
                keyvalues = {
                    _skeleton_name(row.body, for_directory=False): row.buf
                    for row in success_df.itertuples(index=False)
                }
                set_default_dvid_session_timeout(
                    output_cfg['keyvalue']["timeout"],
                    output_cfg['keyvalue']["timeout"]
                )
                out_instance = [output_cfg['keyvalue'][k] for k in ('server', 'uuid', 'instance')]
                total_bytes = int(results_df['buf_size'].sum())
                with mgr_client.access_context(out_instance[0], False, 1, total_bytes):
                    post_keyvalues(*out_instance, keyvalues)

            return results_df.drop(columns=['buf'])

        def process_batch(bodies, batch_id):
            fetched = fetch_sparsevol_batch(bodies)
            results = generate_skeleton_batch(fetched)
            results_df = write_skeletons(batch_id, results)
            results_df['batch_id'] = batch_id
            return batch_id, results_df

        bodies = load_body_list(options["bodies"], False)
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

                if results_df['skeletonize_error'].any():
                    b = results_df.loc[results_df['skeletonize_error'].astype(bool), 'body'].tolist()
                    logger.warning(f"Batch {batch_id}: Failed to skeletonize bodies: {b}")
        finally:
            if all_results_dfs:
                all_results_df = pd.concat(all_results_dfs)
                feather.write_feather(all_results_df, 'skeleton-stats.feather')

                failed_df = all_results_df.query('status != "success"')
                if len(failed_df) > 0:
                    logger.warning(f"{len(failed_df)} skeletons could not be generated. See skeleton-stats.feather")
                    logger.warning("Result summary:\n")
                    logger.warning(f"{all_results_df['status'].value_counts()}")
