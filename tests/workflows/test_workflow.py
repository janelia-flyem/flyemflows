import os
import shutil
import tempfile
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from dvid_resource_manager.client import ResourceManagerClient, TimeoutError

import flyemflows
from flyemflows.util import find_processes
from flyemflows.bin.launchworkflow import launch_workflow

yaml = YAML()
yaml.default_flow_style = False

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')

def checkrun(f):
    """
    Decorator to help you verify that a function was actually executed.
    
    Annotates a function with an attribute 'didrun',
    and only sets it to True if the function is actually called.
    """
    def wrapper(*args, **kwargs):
        wrapper.didrun = True
        return f(*args, **kwargs)
    wrapper.didrun = False
    return wrapper


def test_workflow_environment():
    """
    Users can specify environment variables in their config
    file which will be set in the driver and worker environments.
    
    Make sure those variables are set during the workflow, but not after.
    """
    template_dir = tempfile.mkdtemp(suffix="test-workflow-environment-template")
     
    config = {
        "workflow-name": "workflow",
        "cluster-type": CLUSTER_TYPE,
 
        "environment-variables": {
            "FOO": "BAR"
        }
    }
 
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
 
    orig_environ = os.environ.copy()
    _execution_dir, _workflow = launch_workflow(template_dir, 1, _custom_execute_fn=execute)
    assert execute.didrun
     
    # Environment is restored after execution is finished.
    assert os.environ == orig_environ


def test_resource_manager_on_driver():
    """
    The config can specify a resource manager server address as "driver",
    which means the workflow should launch the resource manager on the scheduler machine.
    Make sure it launches, but is also shut down after the workflow exits.
    """
    template_dir = tempfile.mkdtemp(suffix="test-workflow-environment-template")
     
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
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
 
    @checkrun
    def execute(workflow_inst):
        client = ResourceManagerClient('127.0.0.1', 4000)
        mgr_config = client.read_config()
        assert mgr_config == config["resource-manager"]["config"], \
            "Resource manager config does not match the one in the workflow config"
 
    _execution_dir, _workflow = launch_workflow(template_dir, 1, _custom_execute_fn=execute)
    assert execute.didrun
    
    with pytest.raises(TimeoutError):
        client2 = ResourceManagerClient('127.0.0.1', 4000)
        client2.read_config()


@pytest.fixture(params=[True, False])
def setup_worker_initialization_template(request):
    """
    Setup for test_worker_initialization(), below.
    Parameterized for the "once-per-machine' case and it's opposite.
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
        "cluster-type": "local-cluster",
         
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
    if once_per_machine:
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
 
    _execution_dir, workflow_inst = launch_workflow(template_dir, num_workers, _custom_execute_fn=execute)
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
    shutil.copy(f'{repo_dir}/scripts/worker-dvid/worker-dvid-config.toml',
                f'{template_dir}/worker-dvid-config.toml')
 
    shutil.copy(f'{repo_dir}/scripts/worker-dvid/launch-worker-dvid.sh',
                f'{template_dir}/launch-worker-dvid.sh')
     
    config = {
        "workflow-name": "workflow",
        "cluster-type": "local-cluster",
         
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
 
    _execution_dir, workflow_inst = launch_workflow(template_dir, 1, _custom_execute_fn=execute)
    script_dir = Path(workflow_inst.config['worker-initialization']['script-path']).parent
    assert not is_worker_dvid_running(), \
        ("Worker DVID remained running after the workflow exited."\
         f"Check logs in:\n{script_dir}")


if __name__ == "__main__":
    # I can't run 'local-cluster' tests from within eclipse,
    # hence the '-k not worker_initialization' option
    CLUSTER_TYPE = "synchronous"
    pytest.main(['-s', '--tb=native', '-k', 'not worker_initialization', '--pyargs', 'tests.workflows.test_workflow'])
