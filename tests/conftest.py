import os
import sys
import time
import signal
import subprocess

import pytest
import numpy as np

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
shutdownDelay = 0

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
        stdout = None
        try:
            stdout = dvid_server_proc.communicate(timeout=1.0)
            dvid_server_proc.wait(DVID_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            if stdout:
                print(stdout)
            print("Force-killing dvid")
            dvid_server_proc.send_signal(signal.SIGKILL)
        print("DVID test server is terminated.")

def _launch_dvid_server():
    os.makedirs(DVID_STORE_PATH, exist_ok=True)
    with open(DVID_CONFIG_PATH, 'w') as f:
        f.write(DVID_CONFIG)

    dvid_proc = subprocess.Popen(f'dvid -verbose -fullwrite serve {DVID_CONFIG_PATH}',
                                 shell=True, stdout=subprocess.PIPE) # Hide output ("Sending log messages to...")
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

@pytest.fixture
def disable_auto_retry():
    """
    Tests can include this fixture in their parameter list to ensure timely failures.

    For most tests, auto-retries are not expected to be necessary
    and would only delay failures, slowing down the test
    suite when something is wrong.
    """
    try:
        import flyemflows.util._auto_retry #@UnusedImport
        flyemflows.util._auto_retry.FLYEMFLOWS_DISABLE_AUTO_RETRY = True
        yield
    finally:
        flyemflows.util._auto_retry.FLYEMFLOWS_DISABLE_AUTO_RETRY = False

@pytest.fixture(scope='session')
def random_segmentation():
    """
    Generate a small 'segmentation' with random-ish segment shapes.
    Since this takes a minute to run, we store the results in /tmp
    and only regenerate it if necessary.
    """
    if os.environ.get('TRAVIS', '') == 'true':
        # On Travis-CI, store this test data in a place that gets cached.
        path = '/home/travis/miniconda/test-data/random-test-segmentation.npy'
    else:
        path = '/tmp/random-test-segmentation.npy'

    if os.path.exists(path):
        return np.load(path)

    print("Generating new test segmentation")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    shape = (256,256,256)
    num_seeds = 1000
    seed_coords = tuple(np.random.randint(shape[0], size=(3,num_seeds)))
    seed_vol = np.zeros(shape, dtype=np.uint32)
    seed_vol[seed_coords] = np.arange(1, num_seeds+1)
    
    from vigra.filters import distanceTransform
    from vigra.analysis import watersheds
    
    dt = distanceTransform(seed_vol)
    seg, _maxlabel = watersheds(dt, seeds=seed_vol)

    seg = seg.astype(np.uint64)
    np.save(path, seg)
    return seg

