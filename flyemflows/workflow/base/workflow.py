import os
import sys
import logging
import importlib

from confiddler import load_config

import neuclease
from neuclease.util import Timer

from ...util.cluster_context import ClusterContext
from ...util.dask_util import run_on_each_worker

from .base_schema import BaseSchema
from .contexts import environment_context, LocalResourceManager, WorkerDaemons

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


    @classmethod
    def get_workflow_cls(cls, name, exit_on_error=False):
        """
        Given a workflow name, return the corresponding Workflow subclass type.
        
        Args:
            name:
                Either the name of a 'builtin' workflow like 'CopyGrayscale',
                or a user-defined workflow subclass like
                'mypackage.mymodule.MyWorkflowSubclass'
            
            exit_on_error:
                If True, raise a SystemExit exception if the named
                class can't be found or has an unexpected type.

        Returns:
            A class (type), which is a subclass of Workflow.
        """
        # Avoid circular import
        from .. import BUILTIN_WORKFLOWS
        
        # Is this a fully-qualified custom workflow name?
        if '.' in name:
            *parts, class_name = name.split('.')
            module_name = '.'.join(parts)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            if not issubclass(cls, Workflow):
                msg = f"Class is not a subclass of the Workflow base class: {cls}"
                if exit_on_error:
                    print(msg, file=sys.stderr)
                    sys.exit(1)
                raise RuntimeError(msg)
            return cls
    
        # Is this a built-in workflow name?
        for cls in BUILTIN_WORKFLOWS:
            if name.lower() == cls.__name__.lower():
                return cls
    
        msg = f"Unknown workflow: {name}"
        if exit_on_error:
            print(msg, file=sys.stderr)
            sys.exit(1)
        raise RuntimeError(msg)


    @classmethod
    def load_workflow_config(cls, template_dir):
        """
        Given a workload 'template' directory, load the config from
        ``workflow.yaml`` (including injected defaults).
        
        Args:
            template_dir:
                A template directory containing workflow.yaml
        
        Returns:
            (workflow_cls, config_data)
            A tuple of the workflow class (a type) and the config data (a dict)
        """
        config_path = f'{template_dir}/workflow.yaml'
        
        if not os.path.exists(config_path):
            raise RuntimeError(f"Error: workflow.yaml not found in {template_dir}")
    
        # Determine workflow type and load config
        _cfg = load_config(config_path, {})
        if "workflow-name" not in _cfg:
            raise RuntimeError(f"Workflow config at {config_path} does not specify a workflow-name.")
        
        workflow_cls = Workflow.get_workflow_cls(_cfg['workflow-name'])
        config_data = load_config(config_path, workflow_cls.schema())
        return workflow_cls, config_data
        
    

    def __init__(self, config, num_workers):
        """Initialization of workflow object.

        Args:
            config (dict): loaded config data for workflow, as a dict
            num_workers: How many workers to launch for the job.
                         Note that this is not necessarily the same as
                         the number of nodes (machines),
                         depending on the dask config.
        """
        self.config = config
        neuclease.dvid.DEFAULT_APPNAME = self.config['workflow-name']
        self.num_workers = num_workers

        # Initialized in run()
        self.cc = None

    @property
    def client(self):
        return self.cc and self.cc.client

    def __del__(self):
        # If the cluster is still alive (a debugging feature),
        # kill it now.
        if self.cc:
            self.cc.cleanup()

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
        cluster_type = self.config["cluster-type"]
        max_wait = self.config["cluster-max-wait"]

        # If you're trying to debug a C++ Python extension with AddressSanitizer,
        # uncomment this function call.
        # See developer-examples/ASAN_NOTES.txt for details.
        # self._preload_asan_mac()

        with \
        Timer(f"Running {workflow_name} with {self.num_workers} workers", logger), \
        LocalResourceManager(self.config["resource-manager"]), \
        ClusterContext(cluster_type, self.num_workers, True, max_wait, not kill_cluster) as self.cc, \
        environment_context(self.config["environment-variables"], self), \
        WorkerDaemons(self):
            self.execute()

    def total_cores(self):
        return sum( self.cc.client.ncores().values() )


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
            client = self.cc and self.cc.client
            return run_on_each_worker(func, client, once_per_machine, return_hostnames)

    @staticmethod
    def _preload_asan_mac():
        # See developer-examples/ASAN_NOTES.txt for details.
        CONDA_PREFIX = os.environ["CONDA_PREFIX"]
        asan_lib = f'{CONDA_PREFIX}/lib/clang/11.0.0/lib/darwin/libclang_rt.asan_osx_dynamic.dylib'
        assert os.path.exists(asan_lib)
        os.environ['DYLD_INSERT_LIBRARIES'] = asan_lib
