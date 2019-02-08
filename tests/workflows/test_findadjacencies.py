import os
import copy
import tempfile

import h5py
import numpy as np
import pandas as pd

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchflow import launch_flow

yaml = YAML()
yaml.default_flow_style = False

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')

@pytest.fixture
def setup_findadjacencies():
    template_dir = tempfile.mkdtemp(suffix="findadjacencies-template")
    
    # Create volume
    volume = np.zeros((256,256,256), np.uint64)
    volume[30,  0:192, 0:64] = 1
    volume[31,  0:192, 0:64] = 2

    volume[140,  0:192, 0:64] = 4
    volume[141,  0:192, 0:64] = 3
    
    volume_path = f"{template_dir}/volume.h5"
    with h5py.File(volume_path, 'w') as f:
        f['volume'] = volume
    
    config = {
        "workflow-name": "findadjacencies",
        "cluster-type": CLUSTER_TYPE,
        
        "input": {
            "hdf5": {
                "path": volume_path,
                "dataset": "volume"
            },
            "geometry": {
                "message-block-shape": [64,64,64]
            },
        },
        
        "findadjacencies": {
            "output-table": "output.csv"
        }
    }

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    return template_dir, config, volume


def test_findadjacencies(setup_findadjacencies):
    template_dir, _config, _volume = setup_findadjacencies
    
    execution_dir, workflow = launch_flow(template_dir, 1)
    
    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["findadjacencies"]["output-table"]}')

    #print(output_df.columns)
    #print(output_df)

    label_pairs = output_df[['label_a', 'label_b']].values
    assert 0 not in label_pairs.flat

    label_pairs = list(map(tuple, label_pairs))
    assert (1,2) in label_pairs
    assert (3,4) in label_pairs
    
    assert output_df.query('label_a == 1')['za'].iloc[0] == 30
    assert output_df.query('label_a == 1')['zb'].iloc[0] == 31

    assert output_df.query('label_a == 3')['za'].iloc[0] == 141
    assert output_df.query('label_a == 3')['zb'].iloc[0] == 140 # not 'forward'


def test_findadjacencies_subset_bodies(setup_findadjacencies):
    template_dir, config, _volume = setup_findadjacencies
    
    # Overwrite config with updated settings.
    config = copy.copy(config)
    config["findadjacencies"]["subset-labels"] = [3]
    config["findadjacencies"]["subset-labels-requirement"] = 1

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    execution_dir, workflow = launch_flow(template_dir, 1)
    
    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["findadjacencies"]["output-table"]}')

    #print(output_df.columns)
    #print(output_df)

    label_pairs = output_df[['label_a', 'label_b']].values
    assert 0 not in label_pairs.flat

    label_pairs = list(map(tuple, label_pairs))
    assert (1,2) not in label_pairs
    assert (3,4) in label_pairs
    
    #assert output_df.query('label_a == 1')['z'].iloc[0] == 30
    #assert output_df.query('label_a == 1')['forwardness'].iloc[0] == True

    assert output_df.query('label_a == 3')['za'].iloc[0] == 141
    assert output_df.query('label_a == 3')['zb'].iloc[0] == 140


def test_findadjacencies_subset_edges(setup_findadjacencies):
    template_dir, config, _volume = setup_findadjacencies

    subset_edges = pd.DataFrame([[4,3]], columns=['label_a', 'label_b'])
    subset_edges.to_csv(f'{template_dir}/subset-edges.csv', index=False, header=True)
    
    # Overwrite config with updated settings.
    config = copy.copy(config)
    config["findadjacencies"]["subset-edges"] = 'subset-edges.csv'

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    execution_dir, workflow = launch_flow(template_dir, 1)
    
    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["findadjacencies"]["output-table"]}')

    #print(output_df.columns)
    #print(output_df)

    label_pairs = output_df[['label_a', 'label_b']].values
    assert 0 not in label_pairs.flat

    label_pairs = list(map(tuple, label_pairs))
    assert (1,2) not in label_pairs
    assert (3,4) in label_pairs
    
    #assert output_df.query('label_a == 1')['z'].iloc[0] == 30
    #assert output_df.query('label_a == 1')['forwardness'].iloc[0] == True

    assert output_df.query('label_a == 3')['za'].iloc[0] == 141
    assert output_df.query('label_a == 3')['zb'].iloc[0] == 140


def test_findadjacencies_solid_volume():
    """
    If the volume is solid or empty, an error is raised at the end of the workflow.
    """
    template_dir = tempfile.mkdtemp(suffix="findadjacencies-template")
    
    # Create solid volume
    volume = 99*np.ones((256,256,256), np.uint64)
    volume_path = f"{template_dir}/volume.h5"

    with h5py.File(volume_path, 'w') as f:
        f['volume'] = volume
    
    config = {
        "workflow-name": "findadjacencies",
        "cluster-type": CLUSTER_TYPE,
        
        "input": {
            "hdf5": {
                "path": volume_path,
                "dataset": "volume"
            },
            "geometry": {
                "message-block-shape": [64,64,64]
            },
        },
        
        "findadjacencies": {
            "output-table": "output.csv"
        }
    }

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    with pytest.raises(RuntimeError):
        _execution_dir, _workflow = launch_flow(template_dir, 1)


if __name__ == "__main__":
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_findadjacencies'])
