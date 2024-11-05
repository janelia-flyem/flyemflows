import os
import logging

from neuclease.dvid import (
    set_default_dvid_session_timeout,
    resolve_ref, is_locked,
    fetch_server_info, fetch_repo_instances,
    create_instance, create_tarsupervoxel_instance
)

from ...volumes import DvidVolumeService

logger = logging.getLogger(__name__)

GenericDvidInstanceSchema = {
    "description":
        "Parameters to specify a generic dvid instance (server/uuid/instance).\n"
        "Omitted values will be copied from the input, or given default values.",
    "type": "object",
    "required": ["server", "uuid"],

    # "default": {}, # Must not have default. (Appears below in a 'oneOf' context.)
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
        "instance": {
            "description": "Name of the instance to create",
            "type": "string"
        },
        "sync-to": {
            "description":
                "When creating a tarsupervoxels instance, it should be sync'd to a labelmap instance.\n"
                "Give the instance name here.",
            "type": "string",
            "default": ""
        },
        "timeout": {
            "description": "",
            "type": "number",
            "default": 600.0
        },
        "create-if-necessary": {
            "description":
                "Whether or not to create the instance if it doesn't already exist.\n"
                "If you expect the instance to exist on the server already, leave this\n"
                "set to False to avoid confusion in the case of typos, UUID mismatches, etc.\n",
            "type": "boolean",
            "default": False
        },
    }
}

TarsupervoxelsOutputSchema = {
    "additionalProperties": False,
    "properties": {
        "tarsupervoxels": GenericDvidInstanceSchema
    }
}

KeyvalueOutputSchema = {
    "additionalProperties": False,
    "properties": {
        "keyvalue": GenericDvidInstanceSchema
    }
}

DirectoryOutputSchema = {
    "additionalProperties": False,
    "properties": {
        "directory": {
            "description": "Directory to write supervoxel meshes into.",
            "type": "string",
            # "default": "" # Must not have default. (Appears below in a 'oneOf' context.)
        }
    }
}

DirectoryOfTarfilesOutputSchema = \
{
    "additionalProperties": False,
    "properties": {
        "directory-of-tarfiles": {
            "description":
                "Directory in which to dump batches of supervoxel meshes to.\n"
                "Each batch is written as a single tarfile, suitable for subsequent\n"
                "upload into a DVID tarsupervoxels instance via POST /load\n",
            "type": "string",
            # "default": "" # Must not have default. (Appears below in a 'oneOf' context.)
        }
    }
}


MeshOutputSchema = {
    "oneOf": [
        DirectoryOutputSchema,
        DirectoryOfTarfilesOutputSchema,
        TarsupervoxelsOutputSchema,
        KeyvalueOutputSchema,
    ],
    "default": {"directory": "meshes"}
}


def prepare_mesh_output(output_cfg, output_fmt, input_service):
    """
    If necessary, create the output directory or
    DVID instance so that meshes can be written to it.
    """

    ## directory output
    if 'directory' in output_cfg:
        # Convert to absolute so we can chdir with impunity later.
        output_cfg['directory'] = os.path.abspath(output_cfg['directory'])
        os.makedirs(output_cfg['directory'], exist_ok=True)
        return

    if 'directory-of-tarfiles' in output_cfg:
        # Convert to absolute so we can chdir with impunity later.
        output_cfg['directory-of-tarfiles'] = os.path.abspath(output_cfg['directory-of-tarfiles'])
        os.makedirs(output_cfg['directory-of-tarfiles'], exist_ok=True)
        return

    ##
    ## DVID output (either keyvalue or tarsupervoxels)
    ##
    (instance_type,) = output_cfg.keys()

    set_default_dvid_session_timeout(
        output_cfg[instance_type]["timeout"],
        output_cfg[instance_type]["timeout"]
    )

    server = output_cfg[instance_type]['server']
    uuid = output_cfg[instance_type]['uuid']
    instance = output_cfg[instance_type]['instance']

    # If the output server or uuid is left blank,
    # we assume it should be auto-filled from the input settings.
    if server == "" or uuid == "":
        base_input = input_service.base_service
        if not isinstance(base_input, DvidVolumeService):
            # Can't copy from the input if the input ain't a dvid source
            raise RuntimeError("Output destination server/uuid was left blank.")

        if server == "":
            server = base_input.server
            output_cfg[instance_type]['server'] = server

        if uuid == "":
            uuid = base_input.uuid
            output_cfg[instance_type]['uuid'] = uuid

    # Resolve in case a branch was given instead of a specific uuid
    uuid = resolve_ref(server, uuid)

    if is_locked(server, uuid):
        info = fetch_server_info(server)
        if "Mode" in info and info["Mode"] == "allow writes on committed nodes":
            logger.warning(f"Output is a locked node ({uuid}), but server is in full-write mode. Proceeding.")
        elif os.environ.get("DVID_ADMIN_TOKEN", ""):
            logger.warning(f"Output is a locked node ({uuid}), but you defined DVID_ADMIN_TOKEN. Proceeding.")
        else:
            raise RuntimeError(f"Can't write to node {uuid} because it is locked.")

    if instance_type == 'tarsupervoxels' and not input_is_labelmap_supervoxels(input_service):
        msg = ("You shouldn't write to a tarsupervoxels instance unless "
                "you're reading supervoxels from a labelmap input.\n"
                "Use a labelmap input source, and set supervoxels: true")
        raise RuntimeError(msg)

    existing_instances = fetch_repo_instances(server, uuid)
    if instance in existing_instances:
        # Instance exists -- nothing to do.
        return

    if not output_cfg[instance_type]['create-if-necessary']:
        msg = (f"Output instance '{instance}' does not exist, "
                "and your config did not specify create-if-necessary")
        raise RuntimeError(msg)

    assert instance_type in ('tarsupervoxels', 'keyvalue')

    ## keyvalue output
    if instance_type == "keyvalue":
        create_instance(server, uuid, instance, "keyvalue", tags=["type=meshes"])
        return

    ## tarsupervoxels output
    sync_instance = output_cfg["tarsupervoxels"]["sync-to"]

    if not sync_instance:
        # Auto-fill a default 'sync-to' instance using the input segmentation, if possible.
        base_input = input_service.base_service
        if isinstance(base_input, DvidVolumeService):
            if base_input.instance_name in existing_instances:
                sync_instance = base_input.instance_name

    if not sync_instance:
        msg = ("Can't create a tarsupervoxels instance unless "
                "you specify a 'sync-to' labelmap instance name.")
        raise RuntimeError(msg)

    if sync_instance not in existing_instances:
        msg = ("Can't sync to labelmap instance '{sync_instance}': "
                "it doesn't exist on the output server.")
        raise RuntimeError(msg)

    create_tarsupervoxel_instance(server, uuid, instance, sync_instance, output_fmt)


def input_is_labelmap(input_service):
    return isinstance(input_service.base_service, DvidVolumeService)


def input_is_labelmap_supervoxels(input_service):
    if isinstance(input_service.base_service, DvidVolumeService):
        return input_service.base_service.supervoxels
    return False


def input_is_labelmap_bodies(input_service):
    if isinstance(input_service.base_service, DvidVolumeService):
        return not input_service.base_service.supervoxels
    return False
