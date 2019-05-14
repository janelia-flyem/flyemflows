#!/usr/bin/env python
"""
Initialize a gbucket with a new dvid repo.

Example Usage:

    python init-dvid-bucket.py my-gbucket my-new-repo "This repo is for my stuff"

    python init-dvid-bucket.py --create-bucket my-NEW-gbucket my-new-repo "This repo is for my stuff"

"""
import os
import sys
import time
import textwrap
import argparse
import subprocess

import requests

DVID_CONSOLE_DIR = f'{sys.prefix}/http/dvid-web-console'

#LOG_DIR = os.getcwd() + '/logs'
#if not os.path.exists(LOG_DIR):
#    os.mkdir(LOG_DIR)

LOG_DIR='/tmp/dvid-logs'

def get_toml_text(bucket_name, dvid_console_dir, log_dir):
    return textwrap.dedent("""\
        [server]
        httpAddress = ":8000"
        rpcAddress = ":8001"
        instance_id_gen = "sequential"
        instance_id_start = 100  # new ids start at least from this.
        webClient = "{dvid_console_dir}"
        webDefaultFile = "index.html"

        note = \"""
        {{"source": "gs://{bucket_name}"}}
        \"""

        [logging]
        # We assume each worker is run from a unique directory,
        # so this logfile name need not be unique.
        logfile = "dvid.log"
        max_log_size = 500 # MB
        max_log_age = 30   # days
        [store]
            [store.mutable]
                engine = "gbucket"
                bucket= "{bucket_name}"
        """.format(**locals()))

def main():
    parser = argparse.ArgumentParser(description='Initialize a gbucket with a new dvid repo.')
    parser.add_argument('--create-bucket', action='store_true',
                        help='If provided, the bucket will be created first using gsutil. '
                             'Otherwise, the bucket is assumed to exist.')
    parser.add_argument('--toml-path', default='dvid.toml')
    parser.add_argument('bucket_name')
    parser.add_argument('repo_name')
    parser.add_argument('repo_description')
    args = parser.parse_args()

    try:
        r = requests.get("http://127.0.0.1:8000/api/help")
    except:
        pass
    else:
        sys.stderr.write("ERROR: dvid is already running on 127.0.0.1:8000.\n")
        sys.stderr.write("       Kill that server before running this script!\n")
        sys.exit(1)

    # Strip leading 'gs://', if provided
    if args.bucket_name.startswith('gs://'):
        args.bucket_name = args.bucket_name[len('gs://'):]

    if args.create_bucket:
        subprocess.check_call('gsutil mb -c regional -l us-east4 -p dvid-em gs://{}'.format(args.bucket_name), shell=True)

    print("Writing TOML to {}".format(args.toml_path))
    with open(args.toml_path, 'w') as f_toml:
        f_toml.write(get_toml_text(args.bucket_name, DVID_CONSOLE_DIR, LOG_DIR))

    print("Wrote {}".format(args.toml_path))

    try:
        cmd = 'dvid -verbose serve {toml_path}'.format(toml_path=args.toml_path)
        print(cmd)
        dvid_proc = subprocess.Popen(cmd, shell=True)

        print("Waiting 5 seconds for dvid to start....")
        time.sleep(5.0)

        if dvid_proc.returncode is not None:
            sys.stderr.write("dvid server could not be started!!\n")
            sys.stderr.write("dvid exited with return code: {}\n".format(dvid_proc.returncode))
            sys.exit(1)

        cmd = 'dvid repos new "{}" "{}"'.format(args.repo_name, args.repo_description)
        print(cmd)
        response = subprocess.check_output(cmd, shell=True).strip()
        print(response.decode('utf-8'))
        #repo_uuid = response.split()[-1]
    finally:
        dvid_proc.terminate()
        dvid_proc.wait()


if __name__ == "__main__":
    sys.exit( main() )
