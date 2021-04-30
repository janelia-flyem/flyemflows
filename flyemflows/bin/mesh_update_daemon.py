"""
Daemon process to check for recently modified bodies in a DVID labelmap instance,
and launch the CreateMeshes workflow to generate meshes for those bodies,
via the CreateMeshes 'subset-bodies' option.

Example usage:

    # Prerequisite:
    # Set up password-less ssh access to the submission node,
    # or run this daemon as a foreground process and use --ask-for-password
    # See mesh_update_daemon.py for an example of starting an ssh-agent.

    DAEMON_CMD="mesh_update_daemon update-meshes --reset-kafka-offset --starting-timestamp=2019-12-11 --interval=15 --submission-node=login1.int.janelia.org --email-on-error"

    # Start the job in the background.
    # Since it handles SIGHUP, it should remain running after your terminal closes.
    ${DAEMON_CMD} &>> daemon.log &
"""

#
# # Enable password-less ssh access by creating a private key
# # and then starting an ssh agent as shown below.
#
# function start_agent {
#     echo "Initializing new SSH agent..."
#     # spawn ssh-agent
#     /usr/bin/ssh-agent | sed 's/^echo/#echo/' > "${SSH_ENV}"
#     echo succeeded
#     chmod 600 "${SSH_ENV}"
#     . "${SSH_ENV}" > /dev/null
#     /usr/bin/ssh-add
# }
#
# start_agent
#

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
from getpass import getpass, getuser
from itertools import chain
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('template_dir',
                        help='Path to the a CreateMeshes template directory, i.e. a directory with workflow.yaml and dask-config.yaml. '
                             'Must be accessible both locally and on the submission node.')

    parser.add_argument('--starting-timestamp',
                        help='Only bodies modified after the given timestamp will be processed. '
                             'If not provided, then the starting timestamp is assumed to be the time this daemon was launched.')

    parser.add_argument('--driver-slots', type=int, default=1)
    parser.add_argument('--worker-slots', type=int, default=31)
    parser.add_argument('--ask-for-password', action='store_true')
    parser.add_argument('--conda-path')
    parser.add_argument('--conda-env')
    parser.add_argument('--cwd')
    parser.add_argument('--email-on-error', nargs='?', const=f'{getuser()}@janelia.hhmi.org',
                        help='If provided, an email will be sent to the given address')

    parser.add_argument('--interval', '-i', type=int, default=60,
                        help='How often (in minutes) to check for changed bodies and run the mesh workflow')

    parser.add_argument('--submit-locally', action='store_true',
                        help='If given, run the bsub command to submit the workflow directly on your current machine.')

    parser.add_argument('--submission-node', default='login1.int.janelia.org',
                        help='Which node to launch the workflow from (via ssh).  Ignored if --submit-locally is used.')

    parser.add_argument('--kafka-group-id-suffix',
                        help='Provide this to ensure a unique group id for reading kafka logs. '
                             'Useful if you want to launch two daemons simultaneously with different workflow configs.')

    parser.add_argument('--reset-kafka-offset', action='store_true',
                        help="If True, reset the kafka consumer offset for the daemon's group id, "
                             "forcing the log to be completely reconsumed.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    signal.signal(signal.SIGHUP, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Late imports so --help works quickly
    from requests import HTTPError
    from neuclease import configure_default_logging
    from neuclease.logging_setup import ExceptionLogger
    from neuclease.dvid import reset_kafka_offset, read_labelmap_kafka_df, filter_kafka_msgs_by_timerange

    configure_default_logging()
    logger.info(' '.join(sys.argv))
    logger.info(f"Running as PID {os.getpid()}")

    #
    # Defaults
    #
    if args.starting_timestamp is None:
        args.starting_timestamp = datetime.now()

    if not args.cwd:
        args.cwd = os.getcwd()

    if not args.conda_path:
        r = subprocess.run('which conda', shell=True, capture_output=True, check=True)
        args.conda_path = r.stdout.decode('utf-8').strip()

    if not args.conda_env:
        # TODO: Test that the conda environment works
        args.conda_env = os.environ["CONDA_DEFAULT_ENV"]
        assert args.conda_env != "base", "Don't use the base conda environment!"

    #
    # Initialize ssh connection
    #
    if args.submit_locally:
        c = None
    else:
        c = init_ssh_connection(args.submission_node, args.ask_for_password)

    #
    # Check existence of template directory on submission node
    #
    try:
        run_cmd(c, f'cd {args.cwd} && ls -d {args.template_dir}', log_stdout=False)
    except Exception:
        raise RuntimeError(f"Your template directory {args.template_dir} is not accessible from {args.cwd}")

    #
    # Load workflow config to determine DVID info.
    #
    seg_instance, body_csv = parse_workflow_config(args.template_dir)

    #
    # Kafka setup
    #
    group_id = f'mesh update daemon {seg_instance[0]} {seg_instance[1]} {args.starting_timestamp}'
    if args.kafka_group_id_suffix:
        group_id += ' ' + args.kafka_group_id_suffix

    if args.reset_kafka_offset:
        reset_kafka_offset(*seg_instance, group_id)

    #
    # Main loop
    #
    while True:
        try:
            with ExceptionLogger(logger) as el:
                msgs_df = read_labelmap_kafka_df(*seg_instance, drop_completes=True, group_id=group_id)
                msgs_df = filter_kafka_msgs_by_timerange(msgs_df, args.starting_timestamp)
                extract_body_ids_and_launch(c, args, seg_instance, body_csv, msgs_df)
                need_kafka_reset = False
        except HTTPError:
            msg = ("Failed to process mesh job. (See traceback above.) "
                   "Will reset kafka offset and try again at the next interval.")
            logger.warning(msg)
            send_error_email(el.last_traceback + '\n' + msg, args.email_on_error)
            need_kafka_reset = True

        # TODO: Get feedback on successful/failed runs, and restart failed jobs
        #       (or accumulate their body lists into the next job)
        time.sleep(60*args.interval)

        if need_kafka_reset:
            need_kafka_reset = False
            try:
                reset_kafka_offset(*seg_instance, group_id)
            except HTTPError:
                pass


def parse_workflow_config(template_dir):
    """
    Load workflow.yaml to determine the input volume info and subset-bodies CSV path.
    """
    # Late imports so --help works quickly
    from flyemflows.volumes import VolumeService
    from flyemflows.workflow import Workflow, CreateMeshes

    workflow_cls, workflow_config = Workflow.load_workflow_config(template_dir)

    assert workflow_cls == CreateMeshes
    assert 'dvid' in workflow_config["input"], \
        "This daemon only works with DVID labelmap sources"

    dvid_service = VolumeService.create_from_config(workflow_config["input"]).original_volume_service
    server, _uuid, instance = dvid_service.instance_triple

    # If the config mentions a branch instead of a
    # specific uuid, keep that, not the pre-resolved uuid
    uuid = workflow_config["input"]["dvid"]["uuid"]

    body_csv = workflow_config["createmeshes"]["subset-bodies"]
    assert body_csv == "bodies-to-update.csv", \
        "Your config must have a 'subset-bodies' setting, and it must point to "\
        "bodies-to-update.csv (which will be overwritten by this daemon)"

    seg_instance = (server, uuid, instance)
    return seg_instance, body_csv


def extract_body_ids_and_launch(c, args, seg_instance, body_csv, msgs_df):
    """
    Extract the list of body IDs from the given kafka messages,
    overwrite the body list CSV file in the workflow template directory,
    and submit a cluster job to launch the workflow.
    """
    # Late imports so --help works quickly
    import numpy as np
    import pandas as pd
    from neuclease.dvid import resolve_ref, fetch_mapping, compute_affected_bodies

    if len(msgs_df) == 0:
        return False

    exclude_bodies = load_bad_bodies()

    # If the uuid was specified as a branch,
    # resolve it to a specific uuid now.
    server, uuid, instance = seg_instance
    uuid = resolve_ref(server, uuid)

    # Extract all bodies and supervoxels that have been touched in the kafka log
    new_bodies, changed_bodies, _removed_bodies, new_supervoxels = compute_affected_bodies(msgs_df['msg'])

    # For touched supervoxels, we need to find their mapped bodies.
    sv_split_bodies = set(fetch_mapping(server, uuid, instance, new_supervoxels)) - set([0])

    subset_bodies = set(chain(new_bodies, changed_bodies, sv_split_bodies))
    subset_bodies -= set(exclude_bodies)
    subset_bodies = np.fromiter(subset_bodies, np.uint64)
    subset_bodies = np.sort(subset_bodies).tolist()

    if len(subset_bodies) == 0:
        return False

    # Overwrite the CSV file for the workflow's subset-bodies set.
    pd.Series(subset_bodies, name='body').to_csv(f'{args.template_dir}/{body_csv}', header=True, index=False)

    first_timestamp = msgs_df['timestamp'].iloc[0]
    last_timestamp = msgs_df['timestamp'].iloc[-1]

    logger.info(f"Launching mesh computation for {len(subset_bodies)} bodies, "
                f"modified between [{first_timestamp}] and [{last_timestamp}]")

    # FIXME: Instead of hard-coding -W to one hour, read the template dask-config.yaml
    cmd = (f"source $({args.conda_path} info --base)/bin/activate {args.conda_env} "
           f"&& cd {args.cwd} "
           f"&& bsub -W 01:00 -n {args.driver_slots} -o /dev/null launchflow -n {args.worker_slots} {args.template_dir}")

    run_cmd(c, cmd)
    return True


def handle_signal(signum, frame):
    if signum == signal.SIGHUP:
        logger.info("Received SIGHUP.  Ignoring")
    if signum == signal.SIGTERM:
        raise SystemExit("Received SIGTERM")


def init_ssh_connection(submission_node, ask_for_password):
    from fabric import Connection

    # Test our ability to talk to the submission node
    try:
        c = Connection(submission_node)
        r = c.run('uname -s', hide=True)
    except Exception:
        if ask_for_password:
            logger.info("Couldn't connect to the submission node via passwordless ssh.\n")
            if sys.stdin is None or not sys.stdin.isatty():
                raise RuntimeError("Can't authenticate interactively -- input is not a terminal.")

            c = Connection(submission_node, connect_kwargs={'password': getpass("Enter your password: ")})
            r = c.run('uname -s')
        else:
            msg = ("Couldn't connect to the submission node via passwordless ssh.\n"
                   "Use --ask-for-password for interactive authentication")
            raise RuntimeError(msg)

    assert r.stdout.strip() == 'Linux'
    return c


def run_cmd(c, cmd, log_stdout=True):
    """
    Execute the given shell command on the submission node using the given connection,
    or on the local machine (if c is None).
    """
    if c is None:
        r = subprocess.run(cmd, shell=True, capture_output=True, check=True)
    else:
        r = c.run(cmd, hide=True)

    if log_stdout:
        logger.info(r.stdout.strip())

    if r.stderr.strip():
        logger.error(r.stderr.strip())

    return r.stdout, r.stderr


def send_error_email(error_msg, email_address):
    import socket
    import smtplib
    from email.mime.text import MIMEText

    user = getuser()
    host = socket.gethostname()
    msg = MIMEText(error_msg)
    msg['Subject'] = f'Mesh daemon error'
    msg['From'] = f'mesh_update_daemon <{user}@{host}>'
    msg['To'] = email_address

    try:
        s = smtplib.SMTP('mail.hhmi.org')
        s.sendmail(msg['From'], email_address, msg.as_string())
        s.quit()
    except:
        msg = ("Failed to send error email.  Perhaps your machine "
        "is not configured to send login-less email, which is required for this feature.")
        logger.error(msg)


BAD_BODIES = None


def load_bad_bodies():
    global BAD_BODIES
    if BAD_BODIES is not None:
        return BAD_BODIES

    try:
        BAD_BODIES = []
        # import pandas as pd
        # BAD_BODIES = pd.read_csv('/nrs/flyem/bergs/complete-ffn-agglo/bad-bodies-2019-02-26.csv')['body']
        pass
    except:
        logger.error("Failed to load list of bad bodies to exclude")
        BAD_BODIES = []

    return BAD_BODIES


if __name__ == "__main__":
    main()
