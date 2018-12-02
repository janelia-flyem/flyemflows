import os
import sys
import time
import signal
import subprocess

import pytest

import flyemflows
from neuclease.dvid import create_repo, fetch_repos_info

TEST_DATA_DIR = os.path.dirname(flyemflows.__file__) + '/../tests/test-data'
DVID_STORE_PATH = f'{TEST_DATA_DIR}/dvid-datastore'
DVID_CONFIG_PATH = f'{TEST_DATA_DIR}/dvid-config.toml'

DVID_PORT = 8000
DVID_IP = "127.0.0.1"
DVID_ADDRESS = f"{DVID_IP}:{DVID_PORT}"

DVID_SHUTDOWN_TIMEOUT = 2.0

DVID_CONFIG = f"""\
[server]
httpAddress = ":{DVID_PORT}"
rpcAddress = ":{DVID_PORT+1}"
webClient = "{sys.prefix}/http/dvid-web-console"

[logging]
logfile = "{TEST_DATA_DIR}/dvid.log"
max_log_size = 500 # MB
max_log_age = 30   # days

[store]
    [store.mutable]
    engine = "basholeveldb"
    path = "{DVID_STORE_PATH}"
"""

@pytest.fixture(scope="session")
def setup_dvid_repo():
    """
    Test fixture to launch a dvid server and create an empty repo in it.
    """
    dvid_server_proc, dvid_address = _launch_dvid_server()
    try:
        repo_uuid = _init_test_repo(dvid_address, reuse_existing=False)
        yield dvid_address, repo_uuid
    finally:
        print("\nTerminating DVID test server...")
        dvid_server_proc.send_signal(signal.SIGTERM)
        try:
            dvid_server_proc.wait(DVID_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            print("Force-killing dvid")
            dvid_server_proc.send_signal(signal.SIGKILL)
        print("DVID test server is terminated.")

def _launch_dvid_server():
    os.makedirs(DVID_STORE_PATH, exist_ok=True)
    with open(DVID_CONFIG_PATH, 'w') as f:
        f.write(DVID_CONFIG)

    dvid_proc = subprocess.Popen(f'dvid -verbose -fullwrite serve {DVID_CONFIG_PATH}', shell=True)
    time.sleep(1.0)
    if dvid_proc.poll() is not None:
        raise RuntimeError(f"dvid couldn't be launched.  Exited with code: {dvid_proc.returncode}")
    return dvid_proc, DVID_ADDRESS

def _init_test_repo(dvid_address, reuse_existing=True):
    TEST_REPO_ALIAS = 'neuclease-test'

    if reuse_existing:
        repos_info = fetch_repos_info(dvid_address)
        for repo_uuid, repo_info in repos_info.items():
            if repo_info["Alias"] == TEST_REPO_ALIAS:
                return repo_uuid

    repo_uuid = create_repo(dvid_address, TEST_REPO_ALIAS, 'Test repo for neuclease integration tests')
    return repo_uuid
