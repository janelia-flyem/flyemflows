import os
import sys
import shutil
import tempfile
import functools
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from dvid_resource_manager.client import ResourceManagerClient, TimeoutError

import flyemflows
from flyemflows.workflow import Workflow
from flyemflows.util import find_processes
from flyemflows.bin.launchflow import launch_flow

yaml = YAML()
yaml.default_flow_style = False

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')


def checkrun(f):
    """
    Decorator to help verify that a function was actually executed.
    
    Annotates a function with an attribute 'didrun',
    and only sets it to True if the function is actually called.
    
    Example:
    
        @checkrun
        def myfunc():
            pass
        
        assert (myfunc.didrun == False)
        myfunc()
        assert (myfunc.didrun == True)
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        wrapper.didrun = True
        return f(*args, **kwargs)
    wrapper.didrun = False
    return wrapper


class CustomWorkflow(Workflow):
    @classmethod
    def schema(cls):
        return Workflow.schema()

    @checkrun
    def execute(self):
        pass


def test_workflow_class_discovery():
    """
    Developers can define their own workflow classes in external python packages,
    in which case the workflow-name must be specified as a fully-qualified class name.
    """
    config = {
        "workflow-name": "tests.workflows.test_workflow.CustomWorkflow",
        "cluster-type": CLUSTER_TYPE
    }
 
    template_dir = tempfile.mkdtemp(suffix="test-workflow-discovery-template")
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
 
    _execution_dir, workflow = launch_flow(template_dir, 1)
    assert isinstance(workflow, CustomWorkflow)
    assert workflow.execute.didrun


def test_workflow_environment():
    """
    Users can specify environment variables in their config
    file which will be set in the driver and worker environments.
    
    Make sure those variables are set during the workflow, but not after.
    """
    config = {
        "workflow-name": "workflow",
        "cluster-type": CLUSTER_TYPE,
 
        "environment-variables": {
            "FOO": "BAR",
            "FOO2": "BAR2"
        }
    }
 
    template_dir = tempfile.mkdtemp(suffix="test-workflow-environment-template")
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
 
    @checkrun
    def execute(workflow_inst):
        def _check():
            assert os.environ['FOO'] == "BAR"
            assert os.environ["OMP_NUM_THREADS"] == '1'
            return True
        
        # driver env
        _check()
        
        # worker env
        assert all(workflow_inst.run_on_each_worker(_check).values())
 
    os.environ['FOO'] = 'ORIGINAL_FOO'
    _execution_dir, _workflow = launch_flow(template_dir, 1, _custom_execute_fn=execute)
    assert execute.didrun
     
    # Environment is restored after execution is finished.
    assert os.environ['FOO'] == 'ORIGINAL_FOO'
    assert 'FOO2' not in os.environ


def test_tee_output():
    config = {
        "workflow-name": "workflow",
        "cluster-type": CLUSTER_TYPE,
 
        "environment-variables": {
            "FOO": "BAR"
        }
    }
 
    template_dir = tempfile.mkdtemp(suffix="test-tee-output-template")
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    stdout_msg = "This should appear in output.log\n"
    stderr_msg = "This should also appear in output.log\n"
    traceback_msg = "Tracebacks should appear, too"

    execution_dir = ''
    @checkrun
    def execute(workflow_inst):
        nonlocal execution_dir
        execution_dir = os.getcwd()
        sys.stdout.write(stdout_msg)
        sys.stderr.write(stderr_msg)
        assert False, traceback_msg
    
    with pytest.raises(AssertionError):
        execution_dir, _workflow = launch_flow(template_dir, 1, _custom_execute_fn=execute)

    with open(f'{execution_dir}/output.log', 'r') as f:
        written = f.read()
    
    assert stdout_msg in written
    assert stderr_msg in written
    assert traceback_msg in written


def test_resource_manager_on_driver():
    """
    The config can specify a resource manager server address as "driver",
    which means the workflow should launch the resource manager on the scheduler machine.
    Make sure it launches, but is also shut down after the workflow exits.
    """
    config = {
        "workflow-name": "workflow",
        "cluster-type": CLUSTER_TYPE,
 
        "resource-manager": {
            "server": "driver",
            "port": 4000,
            "config": {
                "read_reqs": 123,
                "read_data": 456,
                "write_reqs": 789,
                "write_data": 321
            }
        }
    }
 
    template_dir = tempfile.mkdtemp(suffix="test-resource-manager-on-driver-template")
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
 
    @checkrun
    def execute(workflow_inst):
        client = ResourceManagerClient('127.0.0.1', 4000)
        mgr_config = client.read_config()
        assert mgr_config == config["resource-manager"]["config"], \
            "Resource manager config does not match the one in the workflow config"
 
    _execution_dir, _workflow = launch_flow(template_dir, 1, _custom_execute_fn=execute)
    assert execute.didrun
    
    # Server should not be running any more after workflow exits.
    with pytest.raises(TimeoutError):
        client2 = ResourceManagerClient('127.0.0.1', 4000)
        client2.read_config()


@pytest.fixture(params=[True, False])
def setup_worker_initialization_template(request):
    """
    Setup for test_worker_initialization(), below.
    Parameterized for the "once-per-machine' case (and its opposite).
    """
    once_per_machine = request.param
    template_dir = tempfile.mkdtemp(suffix="test-worker-initialization")

    worker_script = f"{template_dir}/do-nothing.sh"
    with open(worker_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("sleep 10")
    os.chmod(worker_script, 0o777)
    
    config = {
        "workflow-name": "workflow",
        "cluster-type": CLUSTER_TYPE,
         
        "worker-initialization": {
            "script-path": "do-nothing.sh",
            "only-once-per-machine": once_per_machine,
            "script-args": ["_TEST_SCRIPT_FAKE_ARG_"], # This is just here to make it easy to identify the process
            "launch-delay": 0
        }
    }
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    return template_dir, config, once_per_machine


def test_worker_initialization(setup_worker_initialization_template):
    """
    The config can specify a script to be run on each worker upon cluster initialization.
    This test verifies that it is launched and active while the workflow runs,
    and that it is launched on each worker, or just once per machine, depending on the config.
    """
    template_dir, _config, once_per_machine = setup_worker_initialization_template
    
    num_workers = 2
    if once_per_machine or CLUSTER_TYPE in ("synchronous", "processes"):
        expected_script_count = 1
    else:
        expected_script_count = num_workers
        
    @checkrun
    def execute(workflow_inst):
        script_dir = Path(workflow_inst.config['worker-initialization']['script-path']).parent
        script_count = len(find_processes('_TEST_SCRIPT_FAKE_ARG_'))
        assert script_count > 0, f"Worker script is not running. Check logs in:\n{script_dir}"
        assert script_count <= expected_script_count, f"Worker script started too many times. Check logs in:\n{script_dir}"
        assert script_count == expected_script_count, f"Worker script not started on all workers. Check logs in:\n{script_dir}"
 
    _execution_dir, workflow_inst = launch_flow(template_dir, num_workers, _custom_execute_fn=execute)
    script_dir = Path(workflow_inst.config['worker-initialization']['script-path']).parent
    script_count = len(find_processes('_TEST_SCRIPT_FAKE_ARG_'))

    assert script_count == 0, \
        ("Worker script(s) remained running after the workflow exited."\
         f"Check logs in:\n{script_dir}")
    

def test_worker_dvid_initialization():
    """
    You can provide an initialization script for each worker to call before the workflow starts.
    The most common use-case for such a script is to launch a local dvid server on each worker
    (for posting in parallel to the cloud).
    
    We provide the necessary script for local dvid workers out-of-the-box, in scripts/worker-dvid.
    
    This test verifies that it works.
    """
    repo_dir = Path(flyemflows.__file__).parent.parent
    template_dir = tempfile.mkdtemp(suffix="test-worker-dvid")
 
    # Copy worker script/config into the template
    shutil.copy(f'{repo_dir}/scripts/worker-dvid/dvid.toml',
                f'{template_dir}/dvid.toml')
 
    shutil.copy(f'{repo_dir}/scripts/worker-dvid/launch-worker-dvid.sh',
                f'{template_dir}/launch-worker-dvid.sh')
     
    config = {
        "workflow-name": "workflow",
        "cluster-type": CLUSTER_TYPE,
         
        "worker-initialization": {
            "script-path": "launch-worker-dvid.sh",
            "only-once-per-machine": True,
            "script-args": ["_TEST_SCRIPT_FAKE_ARG_"], # This is just here to make it easy to identify the process
            "launch-delay": 1.0
        }
    }
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    def is_worker_dvid_running():
        return len(find_processes('_TEST_SCRIPT_FAKE_ARG_')) > 0
 
    @checkrun
    def execute(workflow_inst):
        script_dir = Path(workflow_inst.config['worker-initialization']['script-path']).parent
        assert is_worker_dvid_running(), f"Worker DVID is not running. Check logs in:\n{script_dir}"
 
    _execution_dir, workflow_inst = launch_flow(template_dir, 1, _custom_execute_fn=execute)
    script_dir = Path(workflow_inst.config['worker-initialization']['script-path']).parent
    assert not is_worker_dvid_running(), \
        ("Worker DVID remained running after the workflow exited."\
         f"Check logs in:\n{script_dir}")


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")

    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_workflow'])
