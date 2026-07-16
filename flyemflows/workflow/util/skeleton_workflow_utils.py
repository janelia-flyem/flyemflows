import os
import logging

from neuclease.dvid import (
    set_default_dvid_session_timeout,
    resolve_ref, is_locked,
    fetch_server_info, fetch_repo_instances,
    create_instance
)

from ...volumes import DvidVolumeService
from .mesh_workflow_utils import GenericDvidInstanceSchema

logger = logging.getLogger(__name__)

DirectoryOutputSchema = {
    "additionalProperties": False,
    "properties": {
        "directory": {
            "description": "Directory to write skeleton files into.",
            "type": "string",
            # "default": "" # Must not have default. (Appears below in a 'oneOf' context.)
        }
    }
}

DirectoryOfTarfilesOutputSchema = {
    "additionalProperties": False,
    "properties": {
        "directory-of-tarfiles": {
            "description":
                "Directory in which to dump batches of skeletons.\n"
                "Each batch is written as a single tarfile.\n",
            "type": "string",
            # "default": "" # Must not have default. (Appears below in a 'oneOf' context.)
        }
    }
}

KeyvalueOutputSchema = {
    "additionalProperties": False,
    "properties": {
        "keyvalue": GenericDvidInstanceSchema
    }
}


SkeletonOutputSchema = {
    "oneOf": [
        DirectoryOutputSchema,
        DirectoryOfTarfilesOutputSchema,
        KeyvalueOutputSchema,
    ],
    "default": {"directory": "skeletons"}
}


def prepare_skeleton_output(output_cfg, input_service):
    """
    If necessary, create the output directory or DVID keyvalue instance
    so that skeletons can be written to it.

    (Modeled on prepare_mesh_output(), but for the per-body skeleton
    destinations: directory, directory-of-tarfiles, or a DVID keyvalue.)
    """
    ## directory output
    if 'directory' in output_cfg:
        # Convert to absolute so we can chdir with impunity later.
        output_cfg['directory'] = os.path.abspath(output_cfg['directory'])
        os.makedirs(output_cfg['directory'], exist_ok=True)
        return

    if 'directory-of-tarfiles' in output_cfg:
        output_cfg['directory-of-tarfiles'] = os.path.abspath(output_cfg['directory-of-tarfiles'])
        os.makedirs(output_cfg['directory-of-tarfiles'], exist_ok=True)
        return

    ##
    ## DVID keyvalue output
    ##
    (instance_type,) = output_cfg.keys()
    assert instance_type == 'keyvalue'

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

    existing_instances = fetch_repo_instances(server, uuid)
    if instance in existing_instances:
        # Instance exists -- nothing to do.
        return

    if not output_cfg[instance_type]['create-if-necessary']:
        msg = (f"Output instance '{instance}' does not exist, "
               "and your config did not specify create-if-necessary")
        raise RuntimeError(msg)

    create_instance(server, uuid, instance, "keyvalue", tags=["type=skeletons"])
