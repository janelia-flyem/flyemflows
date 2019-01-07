"""
Utilities for the Workflow base class.

The workflow needs to initialize and then tear
down various tools upon launch and exit.

Those initialization/tear-down processes are each encapuslated
as a different context manager defined in this file.

These are not meant to be used by callers other than
the Workflow base class itself.
"""
import os
import sys
import socket
import logging
import getpass
import subprocess
from contextlib import contextmanager

import confiddler.json as json

from ...util import get_localhost_ip_address, kill_if_running

logger = logging.getLogger(__name__)
USER = getpass.getuser()

# driver_ip_addr = '127.0.0.1'
driver_ip_addr = get_localhost_ip_address()

@contextmanager
def environment_context(update_dict):
    """
    Context manager.
    Update the environment variables specified in the given dict
    when the context is entered, and restore the old environment when the context exits.
    
    Note:
        You can modify these or other environment variables while the context is active,
        those modifications will be lost when this context manager exits.
        (Your original environment is restored unconditionally.)
    """
    old_env = os.environ.copy()
    try:
        os.environ.update(update_dict)
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


class LocalResourceManager:
    """
    Context manager.
    
    Based on a workflow's 'resource-manager' config section,
    launch a dvid_resource_manager process on the local machine,
    and shut it down upon exit.
    
    If the 'server' section is not 'driver', then this context manager does nothing.
    
    Note:
        If a resource manager is started, the 'server' configuration
        setting will be overwritten with the local IP address.
    
    Usage:
    
        with LocalResourceManager(workflow.config['resource-manager']):
            # ...workflow executes here...
    """
    
    def __init__(self, resource_manager_config):
        self.resource_manager_config = resource_manager_config
        self.resource_server_process = None

    def __enter__(self):
        """
        Initialize the resource server config members and, if necessary,
        start the resource server process on the driver node.
        
        If the resource server is started locally, the "resource-server"
        setting is OVERWRITTEN in the config data with the driver IP.
        
        Returns:
            The resource server Popen object (if started), or None
        """
        cfg = self.resource_manager_config
        
        server = cfg["server"]
        port = cfg["port"]

        if server == "":
            return None
        
        if port == 0:
            msg = f"You specified a resource server ({server}), but no port"
            raise RuntimeError(msg)
        
        if server != "driver":
            if cfg["config"]:
                msg = ("The resource manager config should only be specified when resource manager 'server' is set to 'driver'."
                       "(If the resource manager server is already running on a different machine, configure it there.)")
                raise RuntimeError(msg)
            return None

        if cfg["config"]:
            tmpdir = f"/tmp/{USER}"
            os.makedirs(tmpdir, exist_ok=True)
            server_config_path = f'{tmpdir}/driver-resource-server-config.json'
            with open(server_config_path, 'w') as f:
                json.dump(cfg["config"], f)
            config_arg = f'--config-file={server_config_path}'
        else:
            config_arg = ''
        
        # Overwrite workflow config data so workers see our IP address.
        cfg["server"] = server = driver_ip_addr

        logger.info(f"Starting resource manager on the driver ({driver_ip_addr}:{port}, a.k.a {socket.gethostname()}:{port})")
        
        python = sys.executable
        cmd = f"{python} {sys.prefix}/bin/dvid_resource_manager {port} {config_arg}"
        self.resource_server_process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, shell=True)
        return self.resource_server_process


    def __exit__(self, *args):
        if self.resource_server_process is None:
            return

        logger.info(f"Terminating resource manager (PID {self.resource_server_process.pid})")
        self.resource_server_process.terminate()
        try:
            self.resource_server_process.wait(10.0)
        except subprocess.TimeoutExpired:
            kill_if_running(self.resource_server_process.pid, 10.0)
