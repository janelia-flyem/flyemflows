import copy
import logging

import numpy as np
import pandas as pd

import dask.config
import distributed

from neuclease.util import tqdm_proxy, Timer
from neuclease.dvid import set_default_dvid_session_timeout
from neuclease.misc.bodymesh import (
    init_mesh_instances, BodyMeshParametersSchema, MeshChunkConfigSchema, update_body_mesh)

from dvid_resource_manager.client import ResourceManagerClient
from flyemflows.workflow.util.config_helpers import BodyListSchema, load_body_list

from ..util.dask_util import as_completed_synchronous
from . import Workflow


logger = logging.getLogger(__name__)

SegmentationDvidInstanceSchema = {
    "type": "object",
    "required": ["server", "uuid", "segmentation-name"],

    "default": {},
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
        "segmentation-name": {
            "description": "Name of the instance to create",
            "type": "string",
            "default": ""
        },
        "timeout": {
            "description": "",
            "type": "number",
            "default": 600.0
        }
    }
}


class ChunkedBodyMeshes(Workflow):

    ChunkedBodyMeshesOptions = {
        "type": "object",
        "additionalProperties": False,
        "default": {},
        "properties": {
            "dvid": SegmentationDvidInstanceSchema,
            "body-meshes": BodyMeshParametersSchema,
            "chunk-meshes": MeshChunkConfigSchema,
            "bodies": BodyListSchema,
            "force-update": {
                "description":
                    "Ignore body meshes on the server and regenerate them from the component chunks.\n",
                "type": "boolean",
                "default": False
            },
            "body-batch-size": {
                "description": "Bodies are processed in batches.\n",
                "type": "integer",
                "minimum": 1,
                "maximum": int(1e6),
                "default": 100,
            },
            "chunk-processes": {
                "description":
                    "How many processes to use within each worker to generate chunk meshes.\n"
                    "The special value -1 means 'schedule chunks onto the general cluster.\n'",
                "type": "integer",
                "minimum": -1,
                # FIXME: Using a magic value (-1) like this is awkward.
                # FIXME: It would be nice to infer the default value from the dask config.
                "default": 0
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "chunkedbodymeshes": ChunkedBodyMeshesOptions
    })

    @classmethod
    def schema(cls):
        return ChunkedBodyMeshes.Schema

    def _init(self):
        cbm_cfg = self.config['chunkedbodymeshes']
        server = cbm_cfg['dvid']['server']
        uuid = cbm_cfg['dvid']['uuid']
        seg_instance = cbm_cfg['dvid']['segmentation-name']

        needs_multiprocessing = (cbm_cfg['chunk-processes'] > 0)
        workers_are_daemon = dask.config.get('distributed.worker.daemon', True)
        if needs_multiprocessing and workers_are_daemon:
            # daemons can't spawn their own child processes, I think.
            msg = ("This workflow uses multiprocessing, so you must configure your dask workers NOT to be daemons.\n"
                    "In your dask-config, set distributed.worker.daemon: false")
            raise RuntimeError(msg)

        timeout = cbm_cfg['dvid']['timeout']
        set_default_dvid_session_timeout(timeout, timeout)
        init_mesh_instances(server, uuid, seg_instance, body=True, chunks=True, sv=False)

    def execute(self):
        self._init()

        cbm_cfg = self.config['chunkedbodymeshes']
        server = cbm_cfg['dvid']['server']
        uuid = cbm_cfg['dvid']['uuid']
        seg_instance = cbm_cfg['dvid']['segmentation-name']
        bodies = load_body_list(cbm_cfg['bodies'], False)
        batch_size = cbm_cfg['body-batch-size']

        resource_config = self.config["resource-manager"]
        resource_mgr = ResourceManagerClient(resource_config["server"], resource_config["port"])

        processes = cbm_cfg['chunk-processes']
        if processes == -1:
            processes = 'dask-worker-client'

        def _update_body_mesh(body):
            update_body_mesh(
                server, uuid, seg_instance,
                body,
                cbm_cfg['body-meshes'],
                cbm_cfg['chunk-meshes'],
                force=cbm_cfg['force-update'],
                processes=processes,
                resource_mgr=resource_mgr
            )
            return body

        with Timer(f"Processing {len(bodies)} bodies", logger):
            task_names = [f'_update_body_mesh-{body}' for body in bodies]
            futures = self.client.map(_update_body_mesh, bodies, key=task_names, priority=0, batch_size=batch_size)
            if hasattr(self.client, 'DEBUG'):
                ac = as_completed_synchronous(futures, with_results=True)
            else:
                ac = distributed.as_completed(futures, with_results=True)

            try:
                completed_bodies = []
                for _, body in tqdm_proxy(ac, total=len(futures)):
                    completed_bodies.append(body)
            finally:
                cb = pd.Series(sorted(completed_bodies), name='body', dtype=np.uint64)
                cb.to_csv('completed-bodies.csv', index=False, header=True)
                if incomplete_bodies := {*bodies} - {*completed_bodies}:
                    logger.warning(f"Did not complete {len(incomplete_bodies)} body meshes (out of {len(bodies)}).")
                    logger.warning("See complete-bodies.csv and incomplete-bodies.csv")
                    pd.Series(sorted(incomplete_bodies), name='body').to_csv('incomplete-bodies.csv', index=False, header=True)
