import os
import pickle
import tempfile
import textwrap
from io import StringIO

import pytest
from ruamel.yaml import YAML

import h5py
import numpy as np

from neuclease.util import  contingency_table

from flyemflows.bin.launchflow import launch_flow

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')


@pytest.fixture
def setup_hdf5_inputs():
    template_dir = tempfile.mkdtemp(suffix="test-contingencytable")
    
    left_vol = np.random.randint(10, size=(256, 256, 256), dtype=np.uint64)
    right_vol = np.random.randint(10, size=(256, 256, 256), dtype=np.uint64)
    
    left_path = f"{template_dir}/left-vol.h5"
    right_path = f"{template_dir}/right-vol.h5"
    
    with h5py.File(left_path, 'w') as f:
        f['volume'] = left_vol

    with h5py.File(right_path, 'w') as f:
        f['volume'] = right_vol
    
    config_text = textwrap.dedent(f"""\
        workflow-name: contingencytable
        cluster-type: {CLUSTER_TYPE}
        
        left-input:
          hdf5:
            path: {left_path}
            dataset: volume
          
          geometry:
            message-block-shape: [256,64,64]

        right-input:
          hdf5:
            path: {right_path}
            dataset: volume
          
          geometry:
            message-block-shape: [256,64,64]
    """)

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)

    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)

    return template_dir, config, left_vol, right_vol


def test_contingencytable(setup_hdf5_inputs):
    template_dir, _config, left_vol, right_vol = setup_hdf5_inputs
    
    expected_table = contingency_table(left_vol, right_vol).reset_index()

    execution_dir, _workflow = launch_flow(template_dir, 1)
    
    with open(f"{execution_dir}/contingency_table.pkl", 'rb') as f:
        output_table = pickle.load(f)

    assert (output_table == expected_table).all().all()


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")
    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_contingencytable']
    #args = ['-k', 'contingencytable'] + args
    pytest.main(args)
