import os
import sys
import time
import socket
import getpass
import logging
import tempfile
import warnings
import subprocess
from collections import defaultdict
from os.path import basename, splitext

import dask
from distributed import Client, LocalCluster, get_worker
from distributed.utils import parse_bytes

import neuclease
from neuclease.util import Timer

import confiddler.json as json
from dvid_resource_manager.server import DEFAULT_CONFIG as DEFAULT_RESOURCE_MANAGER_CONFIG
from ..util import kill_if_running, get_localhost_ip_address, extract_ip_from_link, construct_ganglia_link
from ..util.lsf import construct_rtm_url, get_job_submit_time


logger = logging.getLogger(__name__)

# driver_ip_addr = '127.0.0.1'
driver_ip_addr = get_localhost_ip_address()

USER = getpass.getuser()

# defines workflows that work over DVID
class Workflow(object):
    """
    Base class for all Workflows.

    TODO:
    - Possibly produce profiles of driver functions

    """
    LsfJobSchema = \
    {
        "description": "dask-jobqueue config settings for LSF jobs.\n"
                       "https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html#dask-jobqueue-lsfcluster",
        "type": "object",
        "additionalProperties": True,
        "default": {},
        "properties": {
            "cores": {
                "description": "How many cores for each 'job', (typically an entire node's worth).\n"
                               "The 'workers' (processes) in each job will share these cores amongst them.",
                "type": "integer",
                "minimum": 1,
                "default": 16
            },
            "ncpus": {
                "description": "How many CPUs to reserve for each 'job'.\n"
                               "Typically, this should be the same as 'cores' (which is the default behavior if not specified),\n"
                               "unless you're worried about your RAM usage, in which case you may want it to be higher.\n"
                               "(This setting has no direct effect on dask behavior;\n"
                               "it is solely for fine-tuning resource reservations in the LSF scheduler.)\n",
                "type": "integer",
                "default": -1
            },
            "processes": {
                "description": "How many processes ('workers') per 'job'.\n"
                               "These processes will collectively share the 'cores' you specify for the job.\n"
                               "https://jobqueue.dask.org/en/latest/configuration-setup.html#processes",
                "type": "integer",
                "minimum": 1,
                "default": 1
            },
            "memory": {
                "description": "How much memory to allot to each 'job' (typically an entire\n"
                               "node's worth, assuming the job reserved all CPUs).\n"
                               "This memory will be divided up amongst the workers in the job.\n"
                               "Specified as a string with a suffix for units, e.g. 4GB\n",
                "type": "string",
                "default": "128GB"
            },
            "mem": {
                "description": "How much memory to reserve from LSF for each 'job'.\n"
                               "Typically should be the same as the dask 'memory' setting,\n"
                               "which is the default if not specified here.\n"
                               "(This setting has no direct effect on dask behavior;\n"
                               "it is solely for fine-tuning resource reservations in the LSF scheduler.)\n",
                "type": "string",
                "default": ""
                
            },
            "log-directory": {
                "description": "Where LSF worker logs (from stdout) will be stored.",
                "type": "string",
                "default": "dask-worker-stdout-logs"
            },
            "local-directory": {
                "description": "Where dask should store temporary files when data spills to disk.\n"
                               "Note: Will also be used to configure Python's tempfile.tempdir",
                "type": "string",
                "default": ""
            },
            "walltime": {
                "description": "How much time to give the workers before killing them automatically.\n"
                               "Specified in HH:MM format.\n",
                "type": "string",
                "default": "24:00"
            },
            "death-timeout": {
                "description": "Seconds to wait for a scheduler before closing workers",
                "type": "integer",
                "default": 60
            },
            "name": {
                "description": "The name of the dask worker jobs when submitted to LSF.\n",
                "type": "string",
                "default": "dask-worker"
            }
        }
    }
    
    JobQueueSchema = \
    {
        "description": "dask-jobqueue config settings.",
        "type": "object",
        "additionalProperties": True,
        "default": {},
        "properties": {
            "lsf": LsfJobSchema
        }
    }
    
    DaskConfigSchema = \
    {
        "description": "Dask config values to override the defaults in ~/.config/dask/ or /etc/dask/.\n"
                       "See https://docs.dask.org/en/latest/configuration.html for details.",
        "type": "object",
        "additionalProperties": True,
        "default": {},
        "properties": {
            "jobqueue": JobQueueSchema
        }
    }

    ResourceManagerSchema = \
    {
        "type": "object",
        "default": {},
        "additionalProperties": False,
        "description": "Which resource manager server to use (if any) and how to configure it (if launching on the driver).",
        "properties": {
            "server": {
                "description": "If provided, workflows MAY use this resource server to coordinate competing requests from worker nodes. \n"
                               "Set to the IP address of the (already-running) resource server, or use the special word 'driver' \n"
                               "to automatically start a new resource server on the driver node.",
                "type": "string",
                "default": ""
            },
            "port": {
                "description": "Which port the resource server is running on.  (See description above.)",
                "type": "integer",
                "default": 0
            },
            "config": {
                "type": "object",
                "default": DEFAULT_RESOURCE_MANAGER_CONFIG,
                "additionalProperties": True
            }
        }
    }
    
    WorkerInitSchema = \
    {
        "type": "object",
        "default": {},
        "additionalProperties": False,
        "description": "The given script will be called once per worker node, before the workflow executes.",
        "properties": {
            "script-path": {
                "type": "string",
                "default": ""
            },
            "script-args": {
                "type": "array",
                "items": { "type": "string" },
                "default": []
            },
            "only-once-per-machine": {
                "description": "Depending on your cluster configuration, dask might start multiple workers on a single machine.\n"
                               "Use this setting you only want this initialization script to be run ONCE per machine\n"
                               "(even if there are multiple workers on that machine)",
                "type": "boolean",
                "default": False
            },
            "launch-delay": {
                "description": "By default, wait for the script to complete before continuing.\n"
                               "Otherwise, launch the script asynchronously and then pause for N seconds before continuing.",
                "type": "integer",
                "default": -1 # default: blocking execution
            },
            "log-dir": {
                "type": "string",
                "default": "worker-init-stdout-logs"
            },
            "also-run-on-driver": {
                "description": "Also run this initialization script on the driver machine.\n",
                "type": "boolean",
                "default": False
            }
        }
    }
    
    EnvironmentVariablesSchema = \
    {
        "type": "object",
        "default": {},
        "additionalProperties": { "type": "string" },
        "description": "Extra environment variables to set on the driver and workers.\n"
                       "Some are provided by default, but you may add any others you like.\n",
        "properties": {
            "OMP_NUM_THREADS": {
                "description": "Some pandas and numpy functions will use OpenMP (via MKL or OpenBLAS),\n"
                               "which causes each process to use many threads.\n"
                               "That's bad, since you can end up with N^2 threads on a machine with N cores.\n"
                               "Unless you know what you're doing, it's best to force OpenMP to use only 1 core per process.\n",
                "type": "string",
                "default": "1"
            }
        }
    }

    BaseSchema = \
    {
        "type": "object",
        "description": "Workflow base config",
        "default": {},
        "additionalProperties": False,
        "required": ["workflow-name", "cluster-type"],
        "properties": {
            "workflow-name": {
                "description": "The class name of the workflow which will be executed using this config.",
                "type": "string",
                "minLength": 1
            },
            "cluster-type": {
                "description": "Whether or not to use an LSF cluster or a local cluster.\n"
                               "Choices: lsf, local-cluster, synchronous, processes",
                "type": "string",
                "enum": ["lsf", "local-cluster", "synchronous", "processes"]
                # No default
            },
            "dask-config": DaskConfigSchema,
            "resource-manager": ResourceManagerSchema,
            "worker-initialization": WorkerInitSchema,
            "environment-variables": EnvironmentVariablesSchema
        }
    }
    
    @classmethod
    def schema(cls):
        if cls is Workflow:
            # The Workflow class itself is sometimes "executed" during unit tests,
            # to test generic workflow features (such as worker initialization)
            return Workflow.BaseSchema
        else:
            # Subclasses must implement schema() themselves.
            raise NotImplementedError
    
    def __init__(self, config, num_workers):
        """Initialization of workflow object.

        Args:
            config (dict): loaded config data for workflow, as a dict
            num_workers: How many workers to launch for the job.
                         Note that this is not necessarily the same as the number of nodes (machines),
                         depending on the dask config.
        """
        self.config = config
        neuclease.dvid.DEFAULT_APPNAME = self.config['workflow-name']
        self.num_workers = num_workers
        
        # Initialized in run()
        self.cluster = None
        self.client = None


    def __del__(self):
        # If the cluster is still alive (a debugging feature),
        # kill it now.
        self._cleanup_dask()


    def execute(self):
        if type(self) is Workflow:
            # The Workflow class itself is sometimes "executed" during unit tests,
            # to test generic workflow features (such as worker initialization)
            pass
        else:
            # Subclasses must implement execute() themselves.
            raise NotImplementedError

    
    def run(self, kill_cluster=True):
        """
        Run the workflow by calling the subclass's execute() function
        (with some startup/shutdown steps before/after).
        """
        logger.info(f"Working dir: {os.getcwd()}")
        with Timer(f"Running {self.config['workflow-name']}", logger):
            with Timer("Running initialization steps"):
                self._init_environment_variables()
                resource_server_proc = self._start_resource_server()
                self._init_driver()
                
                # See also: reinitialize_cluster()
                self._init_dask()
                self._run_worker_initializations()

            try:
                self.execute()
            finally:
                sys.stderr.flush()
                
                # See also: reinitialize_cluster()
                self._kill_initialization_procs()
                if kill_cluster:
                    self._cleanup_dask()

                self._kill_resource_server(resource_server_proc)    
                self._restore_original_environment_variables()

                # Only the workflow calls cleanup_faulthandler, once all workers have exited
                # (All workers share the same output file for faulthandler.)
    
                # FIXME
                #cleanup_faulthandler()


    def total_cores(self):
        return sum( self.client.ncores().values() )


    def reinitialize_cluster(self):
        """
        Discard the current cluster (and client) and start up a new one.
        
        Workflows can call this in between 'batches' of work
        to get a new lease on cluster nodes from LSF.
        
        Note: Don't do this while there are tasks in-flight.
        """
        logger.info("Shutting down old cluster")
        self._kill_initialization_procs()
        self._cleanup_dask()
        
        logger.info("Initializing new cluster")
        self._init_dask()
        self._run_worker_initializations()


    def _init_environment_variables(self):
        self._old_env = os.environ.copy()
        os.environ.update(self.config["environment-variables"])


    def _restore_original_environment_variables(self):
        os.environ.clear()
        os.environ.update(self._old_env)


    def _init_driver(self):
        try:
            driver_jobid = os.environ['LSB_JOBID']
        except KeyError:
            pass
        else:
            driver_rtm_url = construct_rtm_url(driver_jobid)
            driver_host = socket.gethostname()
            logger.info(f"Driver host is: {driver_host}")
            logger.info(f"Driver RTM graphs: {driver_rtm_url}")

            start_timestamp = get_job_submit_time()
            ganglia_url = construct_ganglia_link(driver_host, start_timestamp)

            hostgraph_url_path = 'graph-links.txt'
            with open(hostgraph_url_path, 'a') as f:
                f.write(f"{socket.gethostname()} (driver)\n")
                f.write(f"  {ganglia_url}\n")
                f.write(f"  {driver_rtm_url}\n")
            

    def _init_dask(self, wait_for_workers=True):
        # Consider using client.register_worker_callbacks() to configure
        # - faulthandler (later)
        # - excepthook?
        # - (okay, maybe it's just best to put that stuff in __init__.py, like in DSS)
        new_config = dask.config.update(dask.config.config, self.config['dask-config'])
        dask.config.set(new_config)

        if self.config["cluster-type"] == "lsf":
            from dask_jobqueue import LSFCluster #@UnresolvedImport

            ncpus = self.config["dask-config"]["jobqueue"]["lsf"]["ncpus"]
            if ncpus == -1:
                ncpus = self.config["dask-config"]["jobqueue"]["lsf"]["cores"]
                self.config["dask-config"]["jobqueue"]["lsf"]["ncpus"] = ncpus

            mem = self.config["dask-config"]["jobqueue"]["lsf"]["mem"]
            if not mem:
                memory = self.config["dask-config"]["jobqueue"]["lsf"]["memory"]
                mem = parse_bytes(memory)
                self.config["dask-config"]["jobqueue"]["lsf"]["mem"] = mem

            local_dir = self.config["dask-config"]["jobqueue"]["lsf"]["local-directory"]
            if not local_dir:
                user = getpass.getuser()
                local_dir = f"/scratch/{user}"
                self.config["dask-config"]["jobqueue"]["lsf"]["local-directory"] = local_dir
                
                # Set tmp dir, too.
                tempfile.tempdir = local_dir
                os.environ['TMPDIR'] = local_dir # Forked processes will use this for tempfile.tempdir

            # Reconfigure
            new_config = dask.config.update(dask.config.config, self.config['dask-config'])
            dask.config.set(new_config)

            self.cluster = LSFCluster(ip='0.0.0.0')
            self.cluster.scale(self.num_workers)
        elif self.config["cluster-type"] == "local-cluster":
            self.cluster = LocalCluster(ip='0.0.0.0')
            self.cluster.scale(self.num_workers)
        elif self.config["cluster-type"] in ("synchronous", "processes"):
            cluster_type = self.config["cluster-type"]

            # synchronous/processes mode is for testing and debugging only
            assert self.config['dask-config'].get('scheduler', cluster_type) == cluster_type, \
                "Inconsistency between the dask-config and the scheduler you chose."

            if cluster_type == "synchronous":
                ncores = 1
            else:
                import multiprocessing
                ncores = multiprocessing.cpu_count()
            
            dask.config.set(scheduler=self.config["cluster-type"])
            class DebugClient:
                def ncores(self):
                    return {'driver': ncores}
                def close(self):
                    pass
            self.client = DebugClient()
        else:
            assert False, "Unknown cluster type"

        if self.cluster:
            dashboard = self.cluster.dashboard_link
            logger.info(f"Dashboard running on {dashboard}")
            dashboard_ip = extract_ip_from_link(dashboard)
            dashboard = dashboard.replace(dashboard_ip, socket.gethostname())
            logger.info(f"              a.k.a. {dashboard}")
            
            self.client = Client(self.cluster, timeout='60s') # Note: Overrides config value: distributed.comm.timeouts.connect

            # Wait for the workers to spin up.
            with Timer(f"Waiting for {self.num_workers} workers to launch", logger):
                while ( wait_for_workers
                        and self.client.status == "running"
                        and len(self.cluster.scheduler.workers) < self.num_workers ):
                    time.sleep(0.1)

            if wait_for_workers and self.config["cluster-type"] == "lsf":
                self._write_graph_urls('graph-links.txt')


    def _write_graph_urls(self, graph_url_path):
        """
        Write (or append to) the file containing links to the Ganglia and RTM
        hostgraphs for the workers in our cluster.
        
        We emit the following URLs:
            - One Ganglia URL for the combined graphs of all workers
            - One Ganglia URL for each worker
            - One RTM URL for each job (grouped by host)
        """
        assert self.config["cluster-type"] == "lsf"
        job_submit_times = self.run_on_each_worker(get_job_submit_time, True, True)

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
        
        hostgraph_urls = self.run_on_each_worker(construct_rtm_url, False, True)

        # Some workers share the same parent LSF job,
        # and hence have the same hostgraph URL.
        # Don't show duplicate links, but do group the links by host
        # and indicate how many workers are hosted on each node.
        host_url_counts = defaultdict(lambda: defaultdict(lambda: 0))
        for addr, url in hostgraph_urls.items():
            host = addr[len('tcp://'):].split(':')[0]
            host_url_counts[host][url] += 1
        
        with open(graph_url_path, 'a') as f:
            f.write('-'*100 + '\n')
            f.write("Combined Ganglia Graphs:\n")
            f.write(f"  {combined_ganglia_link}\n")
            
            for host, url_counts in host_url_counts.items():
                total_workers = sum(url_counts.values())
                f.write(f"{host} ({total_workers} workers)\n")
                f.write(f"  {host_ganglia_links[host]}\n")
                for url in url_counts.keys():
                    f.write(f"  {url}\n")


    def _cleanup_dask(self):
        if self.client:
            self.client.close()
            self.client = None
        if self.cluster:
            self.cluster.close()
            self.cluster = None


    def _start_resource_server(self):
        """
        Initialize the resource server config members and, if necessary,
        start the resource server process on the driver node.
        
        If the resource server is started locally, the "resource-server"
        setting is OVERWRITTEN in the config data with the driver IP.
        
        Returns:
            The resource server Popen object (if started), or None
        """
        server = self.config["resource-manager"]["server"]
        port = self.config["resource-manager"]["port"]

        if server == "":
            return None
        
        if port == 0:
            msg = f"You specified a resource server ({server}), but no port"
            raise RuntimeError(msg)
        
        if server != "driver":
            if self.config["resource-manager"]["config"]:
                msg = ("The resource manager config should only be specified when resource manager 'server' is set to 'driver'."
                       "(If the resource manager server is already running on a different machine, configure it there.)")
                raise RuntimeError(msg)
            return None

        if self.config["resource-manager"]["config"]:
            tmpdir = f"/tmp/{USER}"
            os.makedirs(tmpdir, exist_ok=True)
            server_config_path = f'{tmpdir}/driver-resource-server-config.json'
            with open(server_config_path, 'w') as f:
                json.dump(self.config["resource-manager"]["config"], f)
            config_arg = f'--config-file={server_config_path}'
        else:
            config_arg = ''
        
        # Overwrite workflow config data so workers see our IP address.
        self.config["resource-manager"]["server"] = server = driver_ip_addr

        logger.info(f"Starting resource manager on the driver ({driver_ip_addr}:{port}, a.k.a {socket.gethostname()}:{port})")
        
        python = sys.executable
        cmd = f"{python} {sys.prefix}/bin/dvid_resource_manager {port} {config_arg}"
        resource_server_process = subprocess.Popen(cmd, stderr=subprocess.STDOUT, shell=True)
        return resource_server_process


    def _kill_resource_server(self, resource_server_proc):
        if resource_server_proc:
            logger.info(f"Terminating resource manager (PID {resource_server_proc.pid})")
            resource_server_proc.terminate()
            try:
                resource_server_proc.wait(10.0)
            except subprocess.TimeoutExpired:
                kill_if_running(resource_server_proc.pid, 10.0)


    def run_on_each_worker(self, func, once_per_machine=False, return_hostnames=True):
        """
        Run the given function once per worker (or once per worker machine).
        Results are returned in a dict of { worker: result }
        
        Args:
            func:
                Must be picklable.
            
            once_per_machine:
                Ensure that the function is only run once per machine,
                even if your cluster is configured to run more than one
                worker on each node.
            
            return_hostnames:
                If True, result keys use hostnames instead of IPs.
        Returns:
            dict:
            { 'ip:port' : result } OR
            { 'hostname:port' : result }
        """
        if self.config["cluster-type"] in ("synchronous", "processes"):
            if return_hostnames:
                results = {f'tcp://{socket.gethostname()}': func()}
            else:
                results = {'tcp://127.0.0.1': func()}
            logger.info(f"Ran {func.__name__} on the driver only")
            return results
        
        all_worker_hostnames = self.client.run(socket.gethostname)
        if not once_per_machine:
            worker_hostnames = all_worker_hostnames

        if once_per_machine:
            machines = set()
            worker_hostnames = {}
            for address, name in all_worker_hostnames.items():
                ip = address.split('://')[1].split(':')[0]
                if ip not in machines:
                    machines.add(ip)
                    worker_hostnames[address] = name
        
        workers = list(worker_hostnames.keys())
        
        with Timer(f"Running {func.__name__} on {len(workers)} workers", logger):
            results = self.client.run(func, workers=workers)
        
        if not return_hostnames:
            return results
    
        final_results = {}
        for address, result in results.items():
            hostname = worker_hostnames[address]
            ip = extract_ip_from_link(address)
            final_results[address.replace(ip, hostname)] = result

        return final_results


    def _run_worker_initializations(self):
        """
        Run an initialization script (e.g. a bash script) on each worker node.
        Returns:
            (worker_init_pids, driver_init_pid), where worker_init_pids is a
            dict of { hostname : PID } containing the PIDs of the init process
            IDs running on the workers.
        """
        init_options = self.config["worker-initialization"]
            
        if not init_options["script-path"]:
            self.worker_init_pids = {}
            self.driver_init_pid = None
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
        
        worker_init_pids = self.run_on_each_worker(launch_init_script,
                                                   init_options["only-once-per-machine"],
                                                   return_hostnames=False)

        driver_init_pid = None
        if init_options["also-run-on-driver"]:
            if self.config["cluster-type"] != "lsf":
                warnings.warn("Warning: You are using a local-cluster, yet your worker initialization specified 'also-run-on-driver'.")
            driver_init_pid = launch_init_script()
        
        if launch_delay > 0:
            logger.info(f"Pausing after launching worker initialization scripts ({launch_delay} seconds).")
            time.sleep(launch_delay)

        self.worker_init_pids = worker_init_pids
        self.driver_init_pid = driver_init_pid


    def _kill_initialization_procs(self):
        """
        Kill any initialization processes (as launched from _run_worker_initializations)
        that might still running on the workers and/or the driver.
        
        If they don't respond to SIGTERM, they'll be force-killed with SIGKILL after 10 seconds.
        """
        launch_delay = self.config["worker-initialization"]["launch-delay"]
        once_per_machine = self.config["worker-initialization"]["only-once-per-machine"]
        
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
            worker_statuses = self.run_on_each_worker(kill_init_proc, once_per_machine, True)
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

