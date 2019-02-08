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

    _ = 0
    #           0 1 2 3  4 5 6 7
    volume = [[[_,_,_,_, _,6,6,6], # 0
               [_,1,1,2, 2,_,6,_], # 1
               [_,1,1,2, 2,_,_,_], # 2
               [_,1,1,2, 8,_,7,7], # 3

               [_,_,_,_, _,_,_,_], # 4
               [_,4,4,4, 4,_,_,_], # 5
               [_,3,3,3, 3,_,_,_], # 6
               [_,_,_,_, _,_,_,_]]]# 7
    #           0 1 2 3  4 5 6 7

    volume = np.array(volume, np.uint64)
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
                "message-block-shape": [4,4,1]
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
    
    assert (output_df.query('label_a == 1')[['za', 'zb']].values[0] == 0).all()
    assert (output_df.query('label_a == 1')[['ya', 'yb']].values[0] == 2).all()
    assert (output_df.query('label_a == 1')[['xa', 'xb']].values[0] == (2,3)).all()

    assert (output_df.query('label_a == 3')[['za', 'zb']].values[0] == 0).all()
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (6,5)).all() # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == 2).all()


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

    label_pairs = output_df[['label_a', 'label_b']].values
    assert 0 not in label_pairs.flat

    label_pairs = list(map(tuple, label_pairs))
    assert (1,2) not in label_pairs
    assert (3,4) in label_pairs

    assert (output_df.query('label_a == 3')[['za', 'zb']].values[0] == 0).all()
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (6,5)).all() # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == 2).all()


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

    label_pairs = output_df[['label_a', 'label_b']].values
    assert 0 not in label_pairs.flat

    label_pairs = list(map(tuple, label_pairs))
    assert (1,2) not in label_pairs
    assert (3,4) in label_pairs
    
    assert (output_df.query('label_a == 3')[['za', 'zb']].values[0] == 0).all()
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (6,5)).all() # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == 2).all()


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


def test_findadjacencies_closest_approach_subset_edges(setup_findadjacencies):
    template_dir, config, _volume = setup_findadjacencies

    subset_edges = pd.DataFrame([[4,3], [7,6]], columns=['label_a', 'label_b'])
    subset_edges.to_csv(f'{template_dir}/subset-edges.csv', index=False, header=True)
    
    # Overwrite config with updated settings.
    config = copy.copy(config)
    config["findadjacencies"]["subset-edges"] = 'subset-edges.csv'
    config["findadjacencies"]["find-closest"] = True

    _impl_test_findadjacencies_closest_approach(template_dir, config)


def test_findadjacencies_closest_approach_subset_labels(setup_findadjacencies):
    template_dir, config, _volume = setup_findadjacencies

    # Overwrite config with updated settings.
    config = copy.copy(config)
    config["findadjacencies"]["subset-labels"] = [3,4,6,7]
    config["findadjacencies"]["find-closest"] = True

    _impl_test_findadjacencies_closest_approach(template_dir, config)


def _impl_test_findadjacencies_closest_approach(template_dir, config):
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    execution_dir, workflow = launch_flow(template_dir, 1)
    
    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["findadjacencies"]["output-table"]}')

    label_pairs = output_df[['label_a', 'label_b']].values
    assert 0 not in label_pairs.flat

    label_pairs = list(map(tuple, label_pairs))
    assert (1,2) not in label_pairs
    assert (3,4) in label_pairs
    assert (6,7) in label_pairs
    
    assert (output_df.query('label_a == 3')[['za', 'zb']].values[0] == 0).all()
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (6,5)).all() # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == 2).all()

    assert (output_df.query('label_a == 6')[['za', 'zb']].values[0] == 0).all()
    assert (output_df.query('label_a == 6')[['ya', 'yb']].values[0] == (1,3)).all() # not 'forward'
    assert (output_df.query('label_a == 6')[['xa', 'xb']].values[0] == 6).all()


if __name__ == "__main__":
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_findadjacencies'])
