import os
import re
import time
import socket
import logging
import subprocess
from collections import defaultdict

import dask
from distributed import Client, LocalCluster

from neuclease.util import Timer

from . import extract_ip_from_link, construct_ganglia_link
from .lsf import construct_rtm_url, get_job_submit_time
from .dask_util import load_and_overwrite_dask_config, update_jobqueue_config_with_defaults, dump_dask_config, DebugClient, run_on_each_worker

logger = logging.getLogger(__name__)
JOBQUEUE_CLUSTERS = ["lsf", "sge", "slurm"]


class ClusterContext:
    """
    Context manager.

    Launches the a dask cluster, with extra logic to log cluster
    links to an output file, dump the full dask config,
    configure worker logging, etc.

    Args:
        cluster_type:
            One of: "local-cluster", "synchronous", "processes", "lsf", "sge", "slurm"

        num_workers:
            How many dask workers to launch.
            Depending on your dask config, this may launch fewer LSF jobs
            (since multiple workers can be configured in a single bsub command).
            It may also reserve more CPU slots than num_workers, since you can
            configure each worker to occupy multiple slots if you want.
            See the dask-jobqueue docs for details.

        wait_for_workers:
            If True, do not enter the context until all workers have started.

        cluster_max_wait:
            If wait_for_workers is True and not all of the clusters' workers can
            be started within the given time (in minutes), then an exception is raised.
            If None, then wait forever.

        defer_cleanup:
            If True, as a special debugging feature, the cluster will not be closed upon context exit.
            The caller is responsible for cleaning up the cluster when you are ready to destroy the
            cluster, e.g. by calling cleanup(), below.
    """
    def __init__(self, cluster_type, num_workers, wait_for_workers=True, cluster_max_wait=60, defer_cleanup=False):
        self.cluster_type = cluster_type
        self.num_workers = num_workers
        self.wait_for_workers = wait_for_workers
        self.cluster_max_wait = cluster_max_wait
        self.defer_cleanup = defer_cleanup

        # Configured in __enter__
        self.client = None

    def __enter__(self):
        try:
            self._init_dask()
            return self
        except BaseException:
            if not self.defer_cleanup:
                self.cleanup()
            raise

    def __exit__(self, *_):
        if not self.defer_cleanup:
            self.cleanup()

    def cleanup(self):
        """
        Close the client and cluster.
        """
        cluster = self.client and self.client.cluster

        if self.client:
            self.client.close()
            self.client = None

        if cluster:
            try:
                cluster.close(timeout=60.0)
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

    def _init_dask(self):
        """
        Starts a dask cluster, according to the cluster type specified in the constructor.
        Sets self.client.
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

        load_and_overwrite_dask_config(self.cluster_type, 'dask-config.yaml', True)
        self._write_driver_graph_urls()

        if self.cluster_type in JOBQUEUE_CLUSTERS:
            update_jobqueue_config_with_defaults(self.cluster_type)

            if self.cluster_type == "lsf":
                from dask_jobqueue import LSFCluster
                cluster = LSFCluster()
            elif self.cluster_type == "sge":
                from dask_jobqueue import SGECluster
                cluster = SGECluster()
            elif self.cluster_type == "slurm":
                from dask_jobqueue import SLURMCluster
                cluster = SLURMCluster()
            else:
                raise AssertionError("Unimplemented jobqueue cluster")

            cluster.scale(self.num_workers)

        elif self.cluster_type == "local-cluster":
            cluster = LocalCluster(self.num_workers, threads_per_worker=1, processes=True)

        elif self.cluster_type in ("synchronous", "processes"):
            cluster = None
            # synchronous/processes mode is for testing and debugging only
            assert dask.config.get('scheduler', self.cluster_type) == self.cluster_type, \
                "Inconsistency between the dask-config and the scheduler you chose."

            dask.config.set(scheduler=self.cluster_type)
            self.client = DebugClient(self.cluster_type)
        else:
            raise AssertionError("Unknown cluster type")

        dump_dask_config('full-dask-config.yaml')

        if cluster:
            dashboard = cluster.dashboard_link
            logger.info(f"Dashboard running on {dashboard}")
            dashboard_ip = extract_ip_from_link(dashboard)
            dashboard = dashboard.replace(dashboard_ip, socket.gethostname())
            logger.info(f"              a.k.a. {dashboard}")

            # Note: Overrides config value: distributed.comm.timeouts.connect
            self.client = Client(cluster, timeout='60s')

            # Wait for the workers to spin up.
            with Timer(f"Waiting for {self.num_workers} workers to launch", logger) as wait_timer:
                while ( self.wait_for_workers
                        and self.client.status == "running"
                        and len(self.client.cluster.scheduler.workers) < self.num_workers ):

                    if wait_timer.seconds > (60 * self.cluster_max_wait):
                        msg = (f"Not all cluster workers could be launched within the "
                                "allotted time ({self.cluster_max_wait} minutes).\n"
                                "Try again or adjust the 'cluster-max-wait' setting.\n")
                        raise RuntimeError(msg)
                    time.sleep(0.1)

            if self.wait_for_workers and self.cluster_type == "lsf":
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
        assert self.cluster_type == "lsf"
        job_submit_times = run_on_each_worker(get_job_submit_time, self.client, True, True)

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

        rtm_urls = run_on_each_worker(construct_rtm_url, self.client, False, True)

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
