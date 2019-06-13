#!/usr/bin/env python3
"""
Entry point to launch a workflow for a given config, from a template directory.

Example usage:

  After you specify your configuration options in my-samplepoints-job/workflow.yaml,
  this command will launch the workflow with 8 workers:
  
    $ launchflow -n 8 my-samplepoints-config-dir

Tip:

  For help understanding a particular workflow's config options,
  try the --dump-default-verbose-yaml option (-v), which will dump out
  a commented default configuration file.

    $ launchflow -v samplepoints | less

  Note: The default config is very verbose.
        If you are satisified with the default setting
        for any option, you may omit it from your config.
        You must only specify those options which have no
        provided default.
"""
import os
import sys
import copy
import time
import shutil
import logging
import argparse
import importlib
import subprocess
from datetime import datetime

import dask.config

import confiddler.json as json
from confiddler import load_config, dump_config, dump_default_config, validate

from neuclease import configure_default_logging
import flyemflows
from flyemflows.util import tee_streams, email_on_exit
from flyemflows.workflow import Workflow, BUILTIN_WORKFLOWS
from flyemflows.workflow.base.dask_schema import DaskConfigSchema

logger = logging.getLogger(__name__)


def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Workflow info
    parser.add_argument('--list-workflows', '-w', action="store_true",
                        help="List all built-in workflows and exit.")

    # Schema/config info
    parser.add_argument('--dump-schema', '-s',
                        help="dump config schema for the given workflow (as json)")
    parser.add_argument('--dump-default-yaml', '-y',
                        help="Dump default config values for the given workflow (as yaml)")
    parser.add_argument('--dump-default-verbose-yaml', '-v',
                        help="Dump default config values for the given workflow (as yaml), commented with field descriptions.")
    parser.add_argument('--dump-complete-config', '-c', action='store_true',
                        help="Load the config from the given template dir, inject default values for missing settings, "
                             "and dump the resulting complete config.  (Do not execute the workflow.)")

    # Launch parameters
    parser.add_argument('--num-workers', '-n', type=int, default=1,
                        help='Number of workers to launch (i.e. each worker is launched with a single bsub command)')
    parser.add_argument('--pause-before-exit', '-p', action='store_true',
                        help="Pause before exiting, to allow you to inspect the dask dashboard before it is shut down.")
    parser.add_argument('template_dir', nargs='?',
                        help='A template directory with a workflow.yaml file '
                             '(and possibly other files/scripts to be used by the workflow.)')
    args = parser.parse_args()

    if args.list_workflows:
        print("Built-in workflows:\n")
        for w in BUILTIN_WORKFLOWS:
            if w is not Workflow:
                print(f"  {w.__name__}")
        print("\nTo run a third-party workflow, use a fully-qualified class name in workflow.yaml.\n")
        print("Example:\n\n  workflow-name: mypackage.mymodule.MyWorkflowSubclass\n")
        sys.exit(0)

    if args.dump_schema:
        workflow_cls = get_workflow_cls(args.dump_schema, True)
        print(json.dumps(workflow_cls.schema(), indent=2))
        sys.exit(0)

    if args.dump_default_yaml:
        workflow_name = args.dump_default_yaml
        workflow_cls = get_workflow_cls(workflow_name, True)
        schema = copy.deepcopy(workflow_cls.schema())
        schema["properties"]["workflow-name"]["default"] = workflow_name.lower()
        dump_default_config(schema, sys.stdout, 'yaml')
        sys.exit(0)

    if args.dump_default_verbose_yaml:
        worfklow_name = args.dump_default_verbose_yaml
        workflow_cls = get_workflow_cls(worfklow_name, True)
        schema = copy.deepcopy(workflow_cls.schema())
        schema["properties"]["workflow-name"]["default"] = worfklow_name.lower()
        dump_default_config(schema, sys.stdout, 'yaml-with-comments')
        sys.exit(0)

    if not args.template_dir:
        print("Error: No config directory specified. Exiting.", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.template_dir):
        print(f"Error: template directory does not exist: {args.template_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.template_dir):
        print(f"Error: Given template directory path is a file, not a directory: {args.template_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dump_complete_config:
        workflow_cls, config_data = _load_workflow_config(args.template_dir)
        dump_config(config_data, sys.stdout)
        sys.exit(0)
    
    # Execute the workflow
    workflow = None
    try:
        _exc_dir, workflow = launch_flow(args.template_dir, args.num_workers, not args.pause_before_exit)
    except:
        if args.pause_before_exit:
            import traceback
            traceback.print_exc()
        else:
            raise
    finally:
        if args.pause_before_exit:
            logger.info("Workflow complete, but pausing now due to --pause-before-exit.  Hit Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

    # Workflow must not be deleted until we're ready to exit.
    if workflow:
        del workflow


def launch_flow(template_dir, num_workers, kill_cluster=True, _custom_execute_fn=None):
    """
    Args:
        template_dir:
            A directory containing 'workflow.yaml'.
            The directory will be copied (with a timestamp in the name),
            and the workflow will be executed from within the new copy.
        
        num_workers:
            The number of dask workers to launch.
        
        kill_cluster:
            If True, kill the cluster once execution is complete.
            If False, leave the cluster running (to allow analysis of the diagnostics dashboard).
        
        _custom_execute_fn:
            Test use only.  Used by unit tests to override the execute() function.
    
    Returns:
        (execution_dir, workflow_inst)
    """
    template_dir = template_dir.rstrip('/')
    workflow_cls, config_data = _load_workflow_config(template_dir)
    
    # Create execution dir (copy of template dir) and make it the CWD
    timestamp = f'{datetime.now():%Y%m%d.%H%M%S}'
    execution_dir = f'{template_dir}-{timestamp}'
    execution_dir = os.path.abspath(execution_dir)
    shutil.copytree(template_dir, execution_dir, symlinks=True)
    os.chmod(f'{execution_dir}/workflow.yaml', 0o444) # read-only
    
    
    # Export conda env for future reference
    try:
        p = subprocess.run('conda env export', shell=True, check=True, stdout=subprocess.PIPE)
        with open(f'{execution_dir}/conda-environment.yml', 'wb') as f:
            f.write(p.stdout)
    except Exception as ex:
        logger.warning(f"Failed to export the conda environment: {ex}")

    # Export bash environment variables for future reference
    with open(f'{execution_dir}/bash-environment.sh', 'w') as f:
        for k,v in os.environ.items():
            f.write(f'{k}={v}\n')
    
    logpath = f'{execution_dir}/output.log'

    # Email is the outer-most context here (not in Workflow.run()),
    # so it will be sent after the log is written.
    with email_on_exit(config_data["exit-email"], config_data["workflow-name"], execution_dir, logpath):
        with tee_streams(logpath):
            logger.info(f"Teeing output to {logpath}")
    
            _load_and_overwrite_dask_config(execution_dir, config_data["cluster-type"])
            _log_flyemflows_version()
            
            # On NFS, sometimes it takes a while for it to flush the file cache to disk,
            # which means your terminal doesn't see the new directory for a minute or two.
            # That's slightly annoying, so let's call sync right away to force the flush.
            #
            # Edit: This seems to cause problems if ANY files are deleted during the sync
            #       (even files not in the execution directory)
            #       Unless I can sort that out (or figure out how to sync ONLY the execution directory),
            #       it's best not to execute this sync.
            #os.system('sync')
            
            workflow_inst = _run_workflow(workflow_cls, execution_dir, config_data, num_workers, kill_cluster, _custom_execute_fn)
            return execution_dir, workflow_inst


def _log_flyemflows_version():
    """
    If we're running flyemflows from a git repo (rather than an installed conda package),
    log the git rev we're using.
    """
    flyemflows_dir = os.path.dirname(flyemflows.__file__)
    flyemflows_git = f'{flyemflows_dir}/../.git'
    if os.path.exists(flyemflows_git):
        env = os.environ.copy()
        env['GIT_DIR'] = flyemflows_git
        r = subprocess.run('git describe', env=env, shell=True, check=True, stdout=subprocess.PIPE)
        git_rev = r.stdout.decode('utf-8').strip()
        logger.info(f"Running flyemflows from git repo at version: {git_rev}")


def _load_workflow_config(template_dir):
    config_path = f'{template_dir}/workflow.yaml'
    
    if not os.path.exists(config_path):
        raise RuntimeError(f"Error: workflow.yaml not found in {template_dir}")

    # Determine workflow type and load config
    _cfg = load_config(config_path, {})
    if "workflow-name" not in _cfg:
        raise RuntimeError(f"Workflow config at {config_path} does not specify a workflow-name.")
    
    workflow_cls = get_workflow_cls(_cfg['workflow-name'])
    config_data = load_config(config_path, workflow_cls.schema())
    return workflow_cls, config_data
    

def _load_and_overwrite_dask_config(execution_dir, cluster_type):
    # Load dask config, inject defaults for (selected) missing entries, and overwrite in-place.
    dask_config_path = os.path.abspath(f'{execution_dir}/dask-config.yaml')
    if os.path.exists(dask_config_path):
        # Check for completely empty dask config file
        from ruamel.yaml import YAML
        yaml = YAML()
        config = yaml.load(open(dask_config_path, 'r'))
        if not config:
            dask_config = {}
            validate(dask_config, DaskConfigSchema, inject_defaults=True)
        else:
            dask_config = load_config(dask_config_path, DaskConfigSchema)
    else:
        dask_config = {}
        validate(dask_config, DaskConfigSchema, inject_defaults=True)

    # Don't pollute the config file with extra jobqueue parameters we aren't using
    if "jobqueue" in dask_config:
        for key in list(dask_config["jobqueue"].keys()):
            if key != cluster_type:
                del dask_config["jobqueue"][key]
    
        if len(dask_config["jobqueue"]) == 0:
            del dask_config["jobqueue"]
    
    dump_config(dask_config, dask_config_path)

    # This environment variable is recognized by dask itself
    os.environ["DASK_CONFIG"] = dask_config_path
    dask.config.paths.append(dask_config_path)
    dask.config.refresh()


def _run_workflow(workflow_cls, execution_dir, config_data, num_workers, kill_cluster=True, _custom_execute_fn=None):
    orig_dir = os.getcwd()
    try:
        os.chdir(execution_dir)
        
        # Export complete config (with defaults injected) for debugging
        dump_config(config_data, 'full-workflow-config.yaml')

        workflow_inst = workflow_cls(config_data, num_workers)
        
        if _custom_execute_fn is not None:
            # For testing only: monkey-patch the execute() function
            workflow_inst.execute = lambda: _custom_execute_fn(workflow_inst)
        
        workflow_inst.run(kill_cluster)
    finally:
        os.chdir(orig_dir)

    return workflow_inst


def get_workflow_cls(name, exit_on_error=False):
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


if __name__ == "__main__":
    sys.exit( main() )
