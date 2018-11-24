"""
Entry point to launch a workflow for a given config, from a template directory.
"""
import os
import sys
import shutil
import logging
import argparse
from datetime import datetime

import confiddler.json as json
from confiddler import dump_default_config, load_config
from flyemflows.workflow import AVAILABLE_WORKFLOWS

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Workflow info
    parser.add_argument('--list-workflows', '-w', action="store_true",
                        help="List all available workflows and exit.")

    # Schema/config info
    parser.add_argument('--dump-schema', '-s',
                        help="dump config schema for the given workflow (as json)")
    parser.add_argument('--dump-default-yaml', '-y',
                        help="Dump default config values for the given workflow (as yaml)")
    parser.add_argument('--dump-default-verbose-yaml', '-v',
                        help="Dump default config values for the given workflow (as yaml), commented with field descriptions.")

    # Launch parameters
    parser.add_argument('--num-workers', '-n', type=int, default=1,
                        help='Number of workers to launch (i.e. each worker is launched with a single bsub command)')
    parser.add_argument('template_dir', nargs='?',
                        help='A template directory with a workflow config file '
                             '(and possibly other files/scripts to be used by the workflow.)')
    args = parser.parse_args()

    if args.list_workflows:
        print( json.dumps( list(AVAILABLE_WORKFLOWS.keys() ), indent=4) )

    if args.dump_schema:
        workflow_cls = get_workflow_cls(args.dump_schema, True)
        print(json.dumps(workflow_cls.schema(), indent=2))
        sys.exit(0)

    if args.dump_default_yaml:
        workflow_cls = get_workflow_cls(args.dump_default_yaml, True)
        dump_default_config(workflow_cls.schema(), sys.stdout, 'yaml')
        sys.exit(0)

    if args.dump_default_verbose_yaml:
        workflow_cls = get_workflow_cls(args.dump_default_verbose_yaml, True)
        dump_default_config(workflow_cls.schema(), sys.stdout, 'yaml-with-comments')
        sys.exit(0)

    # Execute the workflow
    if not args.template_dir:
        print("Error: No config directory specified. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    launch_workflow(args.template_dir, args.num_workers)


def launch_workflow(template_dir, num_workers):    
    config_path = f'{template_dir}/workflow.yaml'
    if not os.path.exists(config_path):
        raise RuntimeError(f"Error: workflow.yaml not found in {template_dir}")

    # Determine workflow type and load config
    _cfg = load_config(config_path, {})
    if "workflow-name" not in _cfg:
        raise RuntimeError(f"Workflow config at {config_path} does not specify a workflow-name.")
    
    workflow_cls = get_workflow_cls(_cfg['workflow-name'])
    config_data = load_config(config_path, workflow_cls.schema())

    # Create execution dir (copy of template dir) and make it the CWD
    timestamp = f'{datetime.now():%Y%m%d.%H%M%S}'
    execution_dir = f'{template_dir}-{timestamp}'
    shutil.copytree(template_dir, execution_dir, symlinks=True)

    workflow_inst = _execute_workflow(workflow_cls, execution_dir, config_data, num_workers)
    return execution_dir, workflow_inst


def _execute_workflow(workflow_cls, execution_dir, config_data, num_workers):
    # This function is separate just for convenient testing.
    orig_dir = os.getcwd()
    try:
        os.chdir(execution_dir)
        workflow_inst = workflow_cls(config_data, num_workers)
        workflow_inst.run()
    finally:
        os.chdir(orig_dir)

    return workflow_inst

def get_workflow_cls(name, exit_on_error=False):
    try:
        return AVAILABLE_WORKFLOWS[name.lower()]
    except KeyError:
        msg = f"Unknown workflow: {name}"
        if exit_on_error:
            print(msg, file=sys.stderr)
            sys.exit(1)
        else:
            raise RuntimeError(msg)


if __name__ == "__main__":
    sys.exit( main() )
