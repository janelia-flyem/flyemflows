import copy
import logging

from neuclease.dvid import set_default_dvid_session_timeout, resolve_ref
from neuclease.misc.multiresmesh import (
    MultiresMeshConfigSchema, init_multires_mesh_instances, update_multires_body_mesh)

from dvid_resource_manager.client import ResourceManagerClient

from .chunkedbodymeshes import (
    SegmentationDvidInstanceSchema, BodyMeshBatchProperties, check_chunk_processes, select_bodies, process_bodies)
from . import Workflow

logger = logging.getLogger(__name__)


class MultiresBodyMeshes(Workflow):
    """
    Generate (or update) neuroglancer multi-resolution body meshes for a list of bodies,
    via neuclease.misc.multiresmesh.update_multires_body_mesh().

    Each body's mesh is assembled from independently generated chunks, and stored with
    enough information to update it later by regenerating only its changed chunks.
    (See neuclease.misc.multiresmesh for details.)

    The same 'mesh-config' can be used by the ngsupport service's /multires-mesh endpoints.
    """

    MultiresBodyMeshesOptions = {
        "type": "object",
        "additionalProperties": False,
        "default": {},
        "properties": {
            "dvid": SegmentationDvidInstanceSchema,
            "mesh-config": MultiresMeshConfigSchema,
            **BodyMeshBatchProperties,
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "multiresbodymeshes": MultiresBodyMeshesOptions
    })

    @classmethod
    def schema(cls):
        return MultiresBodyMeshes.Schema

    def _init(self):
        cfg = self.config['multiresbodymeshes']
        check_chunk_processes(cfg)

        timeout = cfg['dvid']['timeout']
        set_default_dvid_session_timeout(timeout, timeout)

        server = cfg['dvid']['server']
        uuid = cfg['dvid']['uuid']
        seg_instance = cfg['dvid']['segmentation-name']
        init_multires_mesh_instances(server, uuid, seg_instance)

    def execute(self):
        self._init()

        cfg = self.config['multiresbodymeshes']
        server = cfg['dvid']['server']
        uuid = resolve_ref(server, cfg['dvid']['uuid'], expand=True)
        seg_instance = cfg['dvid']['segmentation-name']
        bodies = select_bodies(cfg, server, uuid, seg_instance)

        resource_config = self.config["resource-manager"]
        resource_mgr = ResourceManagerClient(resource_config["server"], resource_config["port"])

        processes = cfg['chunk-processes']
        if processes == -1:
            processes = 'dask-worker-client'

        mesh_config = cfg['mesh-config']

        def _update_body_mesh(body):
            update_multires_body_mesh(
                server, uuid, seg_instance, body, copy.deepcopy(mesh_config),
                force=cfg['force-update'],
                processes=processes,
                resource_mgr=resource_mgr
            )
            return body

        process_bodies(self.client, bodies, _update_body_mesh, cfg)
