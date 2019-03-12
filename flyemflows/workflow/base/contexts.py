"""
Utilities for the Workflow base class implementation.

The workflow needs to initialize and then tear-down
various objects and processes upon launch and exit.

Those initialization/tear-down processes are each encapuslated
as a different context manager, defined in this file.

Note:
    These are considered to be part of the Workflow base class implementation.
    They not meant to be used by callers other than the Workflow base class itself.
    It's just cleaner to implement this functionality as a set of context managers,
    rather than including it in the Workflow class definition.
"""
import os
import re
import sys
import time
import socket
import getpass
import logging
import warnings
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from os.path import splitext, basename


import dask
from distributed import Client, LocalCluster, get_worker

from neuclease.util import Timer
import confiddler.json as json

from ...util import get_localhost_ip_address, kill_if_running, extract_ip_from_link, construct_ganglia_link
from ...util.lsf import construct_rtm_url, get_job_submit_time
from ...util.dask_util import update_lsf_config_with_defaults, dump_dask_config, DebugClient

logger = logging.getLogger(__name__)
USER = getpass.getuser()

# driver_ip_addr = '127.0.0.1'
driver_ip_addr = get_localhost_ip_address()

@contextmanager
def environment_context(update_dict, workflow=None):
    """
    Context manager.
    Update the environment variables specified in the given dict
    when the context is entered, and restore the old environment when the context exits.
    
    If workflow is given, the environment all of the workflow's cluster's workers
    will be updated, too, but their environment won't be cleaned up.
    
    Note:
        You can modify these or other environment variables while the context is active,
        those modifications will be lost when this context manager exits.
        (Your original environment is restored unconditionally.)
    """
    old_env = os.environ.copy()
    try:
        os.environ.update(update_dict)
        if workflow is not None:
            def update_env():
                os.environ.update(update_dict)
            workflow.run_on_each_worker(update_env)
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
            if self.workflow.config["cluster-type"] not in ("lsf", "sge"):
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


class WorkflowClusterContext:
    """
    Context manager.
    
    Launches the cluster for a workflow, based on the workflow config.
    Logs cluster links, etc.
    
    Side effects:
        - Sets workflow.cluster and workflow.client
        - Edits workflow.config
    
    Args:
        workflow_instance:
            A instance of a Workflow subclass
        
        wait_for_workers:
            If True, do not enter the context until all workers have started.
        
        defer_cleanup:
            If True, as a special debugging feature, the cluster will not be closed upon context exit.
            The caller is responsible for cleaning up the cluster when you are ready to destroy the
            cluster, e.g. by calling cleanup(), below.
    """
    def __init__(self, workflow_instance, wait_for_workers=True, defer_cleanup=False):
        self.workflow = workflow_instance
        self.config = self.workflow.config
        self.wait_for_workers = wait_for_workers
        self.defer_cleanup = defer_cleanup
    

    def __enter__(self):
        self._init_dask()
    

    def __exit__(self, *args):
        if not self.defer_cleanup:
            self.cleanup()


    def cleanup(self):
        """
        Close the workflow's client and cluster.
        """
        if self.workflow.client:
            self.workflow.client.close()
            self.workflow.client = None
        if self.workflow.cluster:
            try:
                self.workflow.cluster.close()
            except RuntimeError as ex:
                ## For some reason, sometimes the cluster can't be closed due to some
                ## problem with 'bkill', which fails with an error that looks like the following.
                ## If that happens, try to re-run bkill one more time in the hopes of really
                ## killing the cluster and not leaving lingering workers running.
                ## (This issue has been observed on the Janelia cluster for both dask and spark clusters.)
                ## 
                #     RuntimeError: Command exited with non-zero exit code.
                #     Exit code: 255
                #     Command:
                #     bkill 54421878 54421872 54421877
                #     stdout:
                #      
                #     stderr:
                #     Job <54421878>: Failed in an LSF library call: Slave LIM configuration is not ready yet
                #     Job <54421872>: Failed in an LSF library call: Slave LIM configuration is not ready yet
                #     Job <54421877>: Failed in an LSF library call: Slave LIM configuration is not ready yet
                m = re.search(r'bkill( \d+)+', str(ex))
                if not m:
                    raise

                logger.warn("Failed to kill cluster with bkill, trying one more time...")
                time.sleep(2.0)
                result = subprocess.run(m.group(), shell=True)
                if result.returncode != 0:
                    logger.error("Second attempt to kill the cluster failed!")
                    raise
                
            self.workflow.cluster = None
    
    def _init_dask(self):
        """
        Starts a dask cluster according to the workflow configuration.
        Sets the workflow.cluster and workflow.client members.
        Also writes useful URLs to graph-links.txt.
        
        If the 'cluster-type' is 'synchronous', then the cluster will be
        a special stub class (DebugCluster), which provides dummy
        implementations of a few functions from the DistributedCluster API.
        (Mostly just for convenient unit testing.)
        """
        # Consider using client.register_worker_callbacks() to configure
        # - faulthandler (later)
        # - excepthook?
        # - (okay, maybe it's just best to put that stuff in __init__.py, like in DSS)
        self._write_driver_graph_urls()

        if self.config["cluster-type"] == "lsf":
            from dask_jobqueue import LSFCluster #@UnresolvedImport
            update_lsf_config_with_defaults()
            self.workflow.cluster = LSFCluster(ip='0.0.0.0')
            self.workflow.cluster.scale(self.workflow.num_workers)
        if self.config["cluster-type"] == "sge":
            from dask_jobqueue import SGECluster #@UnresolvedImport
            # FIXME: Do I need to do this for SGE, too?
            #update_lsf_config_with_defaults()
            self.workflow.cluster = SGECluster(ip='0.0.0.0')
            self.workflow.cluster.scale(self.workflow.num_workers)
        elif self.config["cluster-type"] == "local-cluster":
            self.workflow.cluster = LocalCluster(ip='0.0.0.0')
            self.workflow.cluster.scale(self.workflow.num_workers)
        elif self.config["cluster-type"] in ("synchronous", "processes"):
            cluster_type = self.config["cluster-type"]

            # synchronous/processes mode is for testing and debugging only
            assert dask.config.get('scheduler', cluster_type) == cluster_type, \
                "Inconsistency between the dask-config and the scheduler you chose."

            dask.config.set(scheduler=self.config["cluster-type"])
            self.workflow.client = DebugClient(cluster_type)
        else:
            assert False, "Unknown cluster type"

        dump_dask_config('full-dask-config.yaml')

        if self.workflow.cluster:
            dashboard = self.workflow.cluster.dashboard_link
            logger.info(f"Dashboard running on {dashboard}")
            dashboard_ip = extract_ip_from_link(dashboard)
            dashboard = dashboard.replace(dashboard_ip, socket.gethostname())
            logger.info(f"              a.k.a. {dashboard}")
            
            self.workflow.client = Client(self.workflow.cluster, timeout='60s') # Note: Overrides config value: distributed.comm.timeouts.connect

            # Wait for the workers to spin up.
            with Timer(f"Waiting for {self.workflow.num_workers} workers to launch", logger):
                while ( self.wait_for_workers
                        and self.workflow.client.status == "running"
                        and len(self.workflow.cluster.scheduler.workers) < self.workflow.num_workers ):
                    time.sleep(0.1)

            if self.wait_for_workers and self.config["cluster-type"] == "lsf":
                self._write_worker_graph_urls('graph-links.txt')


    def _write_driver_graph_urls(self):
        """
        If we are running on an LSF cluster node,
        write RTM and Ganglia links for the driver
        (i.e. the current machine) to graph-links.txt.
        """
        try:
            driver_jobid = os.environ['LSB_JOBID']
        except KeyError:
            pass
        else:
            driver_rtm_url = construct_rtm_url(driver_jobid)
            driver_host = socket.gethostname()
            logger.info(f"Driver LSB_JOBID is: {driver_jobid}")
            logger.info(f"Driver host is: {driver_host}")
            logger.info(f"Driver RTM graphs: {driver_rtm_url}")

            start_timestamp = get_job_submit_time()
            ganglia_url = construct_ganglia_link(driver_host, start_timestamp)

            hostgraph_url_path = 'graph-links.txt'
            with open(hostgraph_url_path, 'a') as f:
                header = f"=== Client RTM/Ganglia graphs ({socket.gethostname()}) ==="
                f.write(header + "\n")
                f.write("="*len(header) + "\n")
                f.write(f"  {driver_rtm_url}\n")
                f.write(f"  {ganglia_url}\n\n")
            

    def _write_worker_graph_urls(self, graph_url_path):
        """
        Write (or append to) the file containing links to the Ganglia and RTM
        hostgraphs for the workers in our cluster.
        
        We emit the following URLs:
            - One Ganglia URL for the combined graphs of all workers
            - One Ganglia URL for each worker
            - One RTM URL for each job (grouped by host)
        """
        assert self.config["cluster-type"] == "lsf"
        job_submit_times = self.workflow.run_on_each_worker(get_job_submit_time, True, True)

        host_min_submit_times = {}
        for addr, timestamp in job_submit_times.items():
            host = addr[len('tcp://'):].split(':')[0]
            try:
                min_timestamp = host_min_submit_times[host]
                if timestamp < min_timestamp:
                    host_min_submit_times[host] = timestamp
            except KeyError:
                host_min_submit_times[host] = timestamp

        host_ganglia_links = { host: construct_ganglia_link(host, ts) for host,ts in host_min_submit_times.items() }

        all_hosts = list(host_min_submit_times.keys())
        min_timestamp = min(host_min_submit_times.values())
        combined_ganglia_link = construct_ganglia_link(all_hosts, min_timestamp)
        
        rtm_urls = self.workflow.run_on_each_worker(construct_rtm_url, False, True)

        # Some workers share the same parent LSF job,
        # and hence have the same hostgraph URL.
        # Don't show duplicate links, but do group the links by host
        # and indicate how many workers are hosted on each node.
        host_rtm_url_counts = defaultdict(lambda: defaultdict(lambda: 0))
        for addr, url in rtm_urls.items():
            host = addr[len('tcp://'):].split(':')[0]
            host_rtm_url_counts[host][url] += 1
        
        with open(graph_url_path, 'a') as f:
            f.write("=== Worker RTM graphs ===\n")
            f.write("=========================\n")
            for host, url_counts in host_rtm_url_counts.items():
                total_workers = sum(url_counts.values())
                f.write(f"{host} ({total_workers} workers)\n")
                f.write("-------------------------\n")
                for url in url_counts.keys():
                    f.write(f"  {url}\n")
                f.write("-------------------------\n\n")

            f.write('\n')
            f.write("=== Worker Ganglia graphs ===\n")
            f.write("=============================\n\n")
            f.write("All worker hosts:\n")
            f.write("-----------------------------\n")
            f.write(f"  {combined_ganglia_link}\n")

            f.write("=============================\n")
            for host, url_counts in host_rtm_url_counts.items():
                total_workers = sum(url_counts.values())
                f.write(f"{host} ({total_workers} workers)\n")
                f.write("-----------------------------\n")
                f.write(f"  {host_ganglia_links[host]}\n")
                f.write("-----------------------------\n\n")


