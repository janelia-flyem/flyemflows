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
import time
import socket
import getpass
import logging
import warnings
import subprocess
from os.path import splitext, basename
from contextlib import contextmanager

from distributed import get_worker

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
        
        sys.stderr.flush()


class WorkerDaemons:
    """
    Context manager.
    Runs an 'initialization script' or background process on every worker
    (like a daemon, but we'll clean it up when the workflow exits).
    
    See the documentation in the 'worker-initialization' schema for details.
    """
    def __init__(self, workflow_instance):
        self.workflow = workflow_instance
        self.worker_init_pids = {}
        self.driver_init_pid = None
    

    def __enter__(self):
        """
        Run an initialization script (e.g. a bash script) on each worker node.
        Returns:
            (worker_init_pids, driver_init_pid), where worker_init_pids is a
            dict of { hostname : PID } containing the PIDs of the init process
            IDs running on the workers.
        """
        init_options = self.workflow.config["worker-initialization"]
        if not init_options["script-path"]:
            return

        init_options["script-path"] = os.path.abspath(init_options["script-path"])
        init_options["log-dir"] = os.path.abspath(init_options["log-dir"])
        os.makedirs(init_options["log-dir"], exist_ok=True)

        launch_delay = init_options["launch-delay"]
        
        def launch_init_script():
            script_name = splitext(basename(init_options["script-path"]))[0]
            log_dir = init_options["log-dir"]
            hostname = socket.gethostname()
            log_file = open(f'{log_dir}/{script_name}-{hostname}.log', 'w')

            try:
                script_args = [str(a) for a in init_options["script-args"]]
                p = subprocess.Popen( [init_options["script-path"], *script_args],
                                      stdout=log_file, stderr=subprocess.STDOUT )
            except OSError as ex:
                if ex.errno == 8: # Exec format error
                    raise RuntimeError("OSError: [Errno 8] Exec format error\n"
                                       "Make sure your script begins with a shebang line, e.g. !#/bin/bash")
                raise

            if launch_delay == -1:
                p.wait()
                if p.returncode == 126:
                    raise RuntimeError("Permission Error: Worker initialization script is not executable: {}"
                                       .format(init_options["script-path"]))
                assert p.returncode == 0, \
                    "Worker initialization script ({}) failed with exit code: {}"\
                    .format(init_options["script-path"], p.returncode)
                return None

            return p.pid
        
        worker_init_pids = self.workflow.run_on_each_worker( launch_init_script,
                                                             init_options["only-once-per-machine"],
                                                             return_hostnames=False)

        driver_init_pid = None
        if init_options["also-run-on-driver"]:
            if self.workflow.config["cluster-type"] != "lsf":
                warnings.warn("Warning: You are using a local-cluster, yet your worker initialization specified 'also-run-on-driver'.")
            driver_init_pid = launch_init_script()
        
        if launch_delay > 0:
            logger.info(f"Pausing after launching worker initialization scripts ({launch_delay} seconds).")
            time.sleep(launch_delay)

        self.worker_init_pids = worker_init_pids
        self.driver_init_pid = driver_init_pid


    def __exit__(self, *args):
        """
        Kill any initialization processes (as launched from _run_worker_initializations)
        that might still running on the workers and/or the driver.
        
        If they don't respond to SIGTERM, they'll be force-killed with SIGKILL after 10 seconds.
        """
        launch_delay = self.workflow.config["worker-initialization"]["launch-delay"]
        once_per_machine = self.workflow.config["worker-initialization"]["only-once-per-machine"]
        
        if launch_delay == -1:
            # Nothing to do:
            # We already waited for the the init scripts to complete.
            return
        
        worker_init_pids = self.worker_init_pids
        def kill_init_proc():
            try:
                worker_addr = get_worker().address
            except ValueError:
                # Special case for synchronous cluster.
                # See run_on_each_worker
                worker_addr = 'tcp://127.0.0.1'
            
            try:
                pid_to_kill = worker_init_pids[worker_addr]
            except KeyError:
                return None
            else:
                return kill_if_running(pid_to_kill, 10.0)
                
        
        if any(self.worker_init_pids.values()):
            worker_statuses = self.workflow.run_on_each_worker(kill_init_proc, once_per_machine, True)
            for k,_v in filter(lambda k_v: k_v[1] is None, worker_statuses.items()):
                logger.info(f"Worker {k}: initialization script was already shut down.")
            for k,_v in filter(lambda k_v: k_v[1] is False, worker_statuses.items()):
                logger.info(f"Worker {k}: initialization script had to be forcefully killed.")
        else:
            logger.info("No worker init processes to kill")
            

        if self.driver_init_pid:
            kill_if_running(self.driver_init_pid, 10.0)
        else:
            logger.info("No driver init process to kill")

        sys.stderr.flush()

