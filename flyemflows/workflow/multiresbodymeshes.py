import copy
import logging

from requests import HTTPError

from neuclease.dvid import set_default_dvid_session_timeout, resolve_ref
from neuclease.misc.multiresmesh import (
    MultiresMeshConfigSchema, init_multires_mesh_instances, update_multires_body_mesh,
    init_supervoxel_mesh_store, update_multires_sv_mesh)

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

    Alternatively, with 'supervoxels: true', generate supervoxel meshes for a list of
    supervoxels, via neuclease.misc.multiresmesh.update_multires_sv_mesh().

    The same 'mesh-config' can be used by the ngsupport service's /multires-mesh endpoints.
    """

    MultiresBodyMeshesOptions = {
        "type": "object",
        "additionalProperties": False,
        "default": {},
        "properties": {
            "dvid": SegmentationDvidInstanceSchema,
            "mesh-config": MultiresMeshConfigSchema,
            "supervoxels": {
                "description":
                    "If true, the listed IDs ('bodies') are supervoxel IDs, and supervoxel meshes are\n"
                    "generated instead of body meshes (in the mesh-config's 'supervoxel-storage').\n"
                    "Supervoxel meshes are generated only once, so supervoxels whose meshes already\n"
                    "exist are skipped.  Supervoxels which don't exist in the given UUID (e.g. they've\n"
                    "been split) are skipped, too.  (Requires the 'bodies' option.\n"
                    "'force-update' and the mutation-based body selection options are not supported.)\n",
                "type": "boolean",
                "default": False
            },
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

        if not cfg['supervoxels']:
            init_multires_mesh_instances(server, uuid, seg_instance)
            return

        if cfg['bodies-from-mutations'] or cfg['select-bodies-after-derived-update']:
            raise RuntimeError("With 'supervoxels: true', the supervoxels must be listed via 'bodies'.")
        if cfg['force-update']:
            raise RuntimeError("Supervoxel meshes are never regenerated, so 'force-update' isn't supported "
                               "with 'supervoxels: true'.  (Use a different 'supervoxel-storage' instead.)")
        if not cfg['mesh-config']['supervoxel-storage']:
            raise RuntimeError("With 'supervoxels: true', the mesh-config must specify a 'supervoxel-storage'.")

        # Prepare the storage once, rather than in every task.
        # (Also verifies that its existing meshes have the same settings.)
        init_supervoxel_mesh_store(server, uuid, seg_instance, copy.deepcopy(cfg['mesh-config']))

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

        def _update_sv_mesh(sv):
            try:
                update_multires_sv_mesh(
                    server, uuid, seg_instance, sv, copy.deepcopy(mesh_config),
                    processes=processes,
                    resource_mgr=resource_mgr
                )
            except HTTPError as ex:
                if ex.response is None or ex.response.status_code != 404:
                    raise
                logger.warning(f"Supervoxel {sv} doesn't exist in {uuid}; skipping it")
            return sv

        if cfg['supervoxels']:
            process_bodies(self.client, bodies, _update_sv_mesh, cfg)
        else:
            process_bodies(self.client, bodies, _update_body_mesh, cfg)
