import os
import copy
import textwrap
import tempfile
from io import StringIO

import h5py
import numpy as np
import pandas as pd

import pytest
from ruamel.yaml import YAML

from neuclease.dvid import create_labelmap_instance, post_labelmap_voxels
from flyemflows.bin.launchflow import launch_flow
from flyemflows.util import upsample

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
    config = copy.deepcopy(config)
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
    config = copy.deepcopy(config)
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
    config = copy.deepcopy(config)
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


@pytest.fixture(scope="module")
def setup_dvid_segmentation_input(setup_dvid_repo):
    dvid_address, repo_uuid = setup_dvid_repo
 
    # Start with a low-res map of the test data
    # and scale it up 16x to achieve a 128-cube
 
    _ = 0
    #                  0 1 2 3  4 5 6 7
    volume_layout = [[[_,_,_,_, _,6,6,6], # 0
                      [_,1,1,2, 2,_,6,_], # 1
                      [_,1,1,2, 2,_,_,_], # 2
                      [_,1,1,2, 8,_,7,7], # 3

                      [_,_,_,_, _,_,_,_], # 4
                      [_,4,4,4, _,_,_,_], # 5
                      [_,3,3,3, _,5,9,_], # 6
                      [_,_,_,_, _,_,_,_]]]# 7
    #                  0 1 2 3  4 5 6 7
    
    lowres_volume = np.zeros((8,8,8), np.uint64)
    lowres_volume[:] = volume_layout

    volume = upsample(lowres_volume, 16)
    assert volume.shape == (128,128,128)
 
    input_segmentation_name = 'findadjacencies-input'
    create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name)
    post_labelmap_voxels(dvid_address, repo_uuid, input_segmentation_name, (0,0,0), volume)
    
    template_dir = tempfile.mkdtemp(suffix="findadjacencies-from-dvid")
 
    config_text = textwrap.dedent(f"""\
        workflow-name: findadjacencies
        cluster-type: {CLUSTER_TYPE}
         
        input:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {input_segmentation_name}
            supervoxels: true
 
        findadjacencies:
          output-table: output.csv
          find-closest: true
    """)
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)
 
    return template_dir, config, volume, dvid_address, repo_uuid


def test_findadjacencies_from_dvid_sparse_labels(setup_dvid_segmentation_input):
    template_dir, config, _volume, _dvid_address, _repo_uuid = setup_dvid_segmentation_input
    config = copy.deepcopy(config)
    config["findadjacencies"]["subset-labels"] = [1,2,3,4,6,7,8]
    _impl_test_findadjacencies_from_dvid_sparse(template_dir, config)


def test_findadjacencies_from_dvid_sparse_edges(setup_dvid_segmentation_input):
    template_dir, config, _volume, _dvid_address, _repo_uuid = setup_dvid_segmentation_input

    subset_edges = pd.DataFrame([[1,2], [4,3], [7,6]], columns=['label_a', 'label_b'])
    subset_edges.to_csv(f'{template_dir}/subset-edges.csv', index=False, header=True)
    
    # Overwrite config with updated settings.
    config = copy.deepcopy(config)
    config["findadjacencies"]["subset-edges"] = 'subset-edges.csv'

    _impl_test_findadjacencies_from_dvid_sparse(template_dir, config)
    

def _impl_test_findadjacencies_from_dvid_sparse(template_dir, config):
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    execution_dir, workflow = launch_flow(template_dir, 1)
    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["findadjacencies"]["output-table"]}')

    label_pairs = output_df[['label_a', 'label_b']].values
    assert 0 not in label_pairs.flat

    label_pairs = list(map(tuple, label_pairs))
    assert (1,2) in label_pairs
    assert (3,4) in label_pairs
    assert (6,7) in label_pairs
    
    assert (output_df.query('label_a == 3')[['za', 'zb']].values[0] == 31).all()
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (6*16, 6*16-1)).all() # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == 2.5*16-1).all()

    # The Z and X locations here are a little hard to test, since several voxels are tied.
    #assert (output_df.query('label_a == 6')[['za', 'zb']].values[0] == 31).all()
    assert (output_df.query('label_a == 6')[['ya', 'yb']].values[0] == (2*16-1, 3*16)).all() # not 'forward'
    assert (output_df.query('label_a == 6')[['xa', 'xb']].values[0] == 6*16).all()
    

if __name__ == "__main__":
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_findadjacencies'])
