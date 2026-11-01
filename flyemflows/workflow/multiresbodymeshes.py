import copy
import logging

from requests import HTTPError

from neuclease.dvid import set_default_dvid_session_timeout, resolve_ref
from neuclease.misc.multiresmesh import (
    MultiresMeshConfigSchema, init_multires_mesh_instances, update_multires_body_mesh,
    init_supervoxel_mesh_store, update_multires_sv_mesh)

from dvid_resource_manager.client import ResourceManagerClient
from flyemflows.workflow.util.config_helpers import BodyListSchema, load_body_list

from .chunkedbodymeshes import (
    SegmentationDvidInstanceSchema, BodyMeshBatchProperties, check_chunk_processes, select_bodies, process_bodies)
from . import Workflow

logger = logging.getLogger(__name__)

MultiresDvidInstanceSchema = copy.deepcopy(SegmentationDvidInstanceSchema)
MultiresDvidInstanceSchema["properties"]["supervoxels"] = {
    "description":
        "If true, generate supervoxel meshes (for the listed 'supervoxels') instead of body meshes.\n"
        "They're stored in the mesh-config's 'supervoxel-storage'.\n",
    "type": "boolean",
    "default": False
}

SupervoxelListSchema = copy.deepcopy(BodyListSchema)
SupervoxelListSchema["description"] = (
    "If dvid: supervoxels is true, the supervoxels whose meshes should be generated:\n"
    "a list of supervoxel IDs, or a path to a CSV file with the list (its 'sv' column, if it has several).\n"
    "Supervoxel meshes are generated only once, so supervoxels whose meshes already exist are skipped,\n"
    "as are supervoxels which don't exist in the given UUID (e.g. because they've been split).\n"
)


class MultiresBodyMeshes(Workflow):
    """
    Generate (or update) neuroglancer multi-resolution body meshes for a list of bodies,
    via neuclease.misc.multiresmesh.update_multires_body_mesh().

    Each body's mesh is assembled from independently generated chunks, and stored with
    enough information to update it later by regenerating only its changed chunks.
    (See neuclease.misc.multiresmesh for details.)

    Alternatively, with 'dvid: supervoxels: true', generate supervoxel meshes for a list of
    supervoxels ('supervoxels'), via neuclease.misc.multiresmesh.update_multires_sv_mesh().

    The same 'mesh-config' can be used by the ngsupport service's /multires-mesh endpoints.
    """

    MultiresBodyMeshesOptions = {
        "type": "object",
        "additionalProperties": False,
        "default": {},
        "properties": {
            "dvid": MultiresDvidInstanceSchema,
            "mesh-config": MultiresMeshConfigSchema,
            "supervoxels": SupervoxelListSchema,
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

        if not cfg['dvid']['supervoxels']:
            if not _is_empty_list(cfg['supervoxels']):
                raise RuntimeError("A 'supervoxels' list was given, but 'dvid: supervoxels' isn't true.  "
                                   "(For body meshes, list the bodies via 'bodies'.)")
            init_multires_mesh_instances(server, uuid, seg_instance)
            return

        if not _is_empty_list(cfg['bodies']) or cfg['bodies-from-mutations'] or cfg['select-bodies-after-derived-update']:
            raise RuntimeError("With 'dvid: supervoxels: true', list the supervoxels via 'supervoxels' "
                               "(and leave the body selection options unset).")
        if _is_empty_list(cfg['supervoxels']):
            raise RuntimeError("With 'dvid: supervoxels: true', the supervoxels must be listed via 'supervoxels'.")
        if cfg['force-update']:
            raise RuntimeError("Supervoxel meshes are never regenerated, so 'force-update' isn't supported "
                               "with 'dvid: supervoxels: true'.  (Use a different 'supervoxel-storage' instead.)")
        if not cfg['mesh-config']['supervoxel-storage']:
            raise RuntimeError("With 'dvid: supervoxels: true', the mesh-config must specify a 'supervoxel-storage'.")

        # Prepare the storage once, rather than in every task.
        # (Also verifies that its existing meshes have the same settings.)
        init_supervoxel_mesh_store(server, uuid, seg_instance, copy.deepcopy(cfg['mesh-config']))

    def execute(self):
        self._init()

        cfg = self.config['multiresbodymeshes']
        server = cfg['dvid']['server']
        uuid = resolve_ref(server, cfg['dvid']['uuid'], expand=True)
        seg_instance = cfg['dvid']['segmentation-name']

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

        if cfg['dvid']['supervoxels']:
            supervoxels = load_body_list(cfg['supervoxels'], True)
            process_bodies(self.client, supervoxels, _update_sv_mesh, cfg)
        else:
            bodies = select_bodies(cfg, server, uuid, seg_instance)
            process_bodies(self.client, bodies, _update_body_mesh, cfg)


def _is_empty_list(id_list):
    """True if a BodyListSchema setting is unset (an empty list or an empty path)."""
    return (isinstance(id_list, list) and len(id_list) == 0) or id_list == ""
