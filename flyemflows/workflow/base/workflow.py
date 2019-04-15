import os
import logging

import neuclease
from neuclease.util import Timer

from .base_schema import BaseSchema
from .contexts import environment_context, LocalResourceManager, WorkerDaemons, WorkflowClusterContext
from ...util.dask_util import run_on_each_worker

logger = logging.getLogger(__name__)

# defines workflows that work over DVID
class Workflow(object):
    """
    Base class for all Workflows.

    TODO:
    - Possibly produce profiles of driver functions

    """
    
    @classmethod
    def schema(cls):
        if cls is Workflow:
            # The Workflow class itself is sometimes "executed" during unit tests,
            # to test generic workflow features (such as worker initialization)
            return BaseSchema
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
        
        # Initialized in run(), specifically by the WorkflowClusterContext
        self.cluster = None
        self.client = None


    def __del__(self):
        # If the cluster is still alive (a debugging feature),
        # kill it now.
        if self.client:
            self.client.close()
            self.client = None
        if self.cluster:
            self.cluster.close()
            self.cluster = None


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

        # The execute() function is run within these nested contexts.
        # See contexts.py
        workflow_name = self.config['workflow-name']
        with Timer(f"Running {workflow_name} with {self.num_workers} workers", logger), \
             LocalResourceManager(self.config["resource-manager"]), \
             WorkflowClusterContext(self, True, not kill_cluster), \
             environment_context(self.config["environment-variables"], self), \
             WorkerDaemons(self):
                self.execute()


    def total_cores(self):
        return sum( self.client.ncores().values() )


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
            return run_on_each_worker(func, None, once_per_machine, return_hostnames)
        else:
            return run_on_each_worker(func, self.client, once_per_machine, return_hostnames)
    
