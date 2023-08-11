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
from flyemflows.workflow.findadjacencies import append_group_col

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
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (6,5)).all()  # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == 2).all()

    # Check CC groups
    cc_sets = set()
    for _cc, cc_df in output_df.groupby('group_cc'):
        cc_set = frozenset(cc_df[['label_a', 'label_b']].values.flat)
        cc_sets.add(cc_set)
    assert set(cc_sets) == { frozenset({1,2,8}), frozenset({4,3}) }


# For now, subset-labels-requirement must be 2.  (1 is not supported any more)
@pytest.mark.xfail
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

    assert (output_df.query('label_a == 3')[['za', 'zb']].values[0] == (0,0)).all()
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (6,5)).all() # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == (2,2)).all()

    # Check CC groups
    cc_sets = set()
    for _cc, cc_df in output_df.groupby('group_cc'):
        cc_set = frozenset(cc_df[['label_a', 'label_b']].values.flat)
        cc_sets.add(cc_set)
    assert set(cc_sets) == { frozenset({4,3}) }


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
    
    assert (output_df.query('label_a == 3')[['za', 'zb']].values[0] == (0,0)).all()
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (6,5)).all() # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == (2,2)).all()


def test_findadjacencies_subset_edges_and_nudge(setup_findadjacencies):
    template_dir, config, _volume = setup_findadjacencies

    subset_edges = pd.DataFrame([[4,3]], columns=['label_a', 'label_b'])
    subset_edges.to_csv(f'{template_dir}/subset-edges.csv', index=False, header=True)

    # Overwrite config with updated settings.
    config = copy.deepcopy(config)
    config["findadjacencies"]["subset-edges"] = 'subset-edges.csv'
    config["findadjacencies"]["nudge-points-apart"] = True

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

    assert (output_df.query('label_a == 3')[['za', 'zb']].values[0] == (0,0)).all()
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (6,5)).all() # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == (2,2)).all()


# This used to be true, but at some point I removed the error.
# I'm considering adding it back, so I'll keep this test for now.
@pytest.mark.xfail
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

    with pytest.raises(RuntimeError, match=r".*No edges were found.*"):
        _execution_dir, _workflow = launch_flow(template_dir, 1)


def test_findadjacencies_closest_approach_subset_edges(setup_findadjacencies):
    template_dir, config, _volume = setup_findadjacencies

    subset_edges = pd.DataFrame([[4,3], [7,6], [6,8]], columns=['label_a', 'label_b'])
    subset_edges.to_csv(f'{template_dir}/subset-edges.csv', index=False, header=True)
    
    # Overwrite config with updated settings.
    config = copy.deepcopy(config)
    config["findadjacencies"]["subset-edges"] = 'subset-edges.csv'
    config["findadjacencies"]["find-closest-using-scale"] = 0

    _output_df = _impl_test_findadjacencies_closest_approach(template_dir, config)



def test_findadjacencies_closest_approach_subset_labels(setup_findadjacencies):
    template_dir, config, _volume = setup_findadjacencies

    # Overwrite config with updated settings.
    config = copy.copy(config)
    config["findadjacencies"]["subset-labels"] = [3,4,6,7,8]
    config["findadjacencies"]["find-closest-using-scale"] = 0

    output_df = _impl_test_findadjacencies_closest_approach(template_dir, config)
    
    # Check CC groups
    cc_sets = set()
    for _cc, cc_df in output_df.groupby('group_cc'):
        cc_set = frozenset(cc_df[['label_a', 'label_b']].values.flat)
        cc_sets.add(cc_set)
    
    assert set(cc_sets) == { frozenset(s) for s in [{6,7,8}, {3,4}] }


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
    assert (6,7) in label_pairs
    
    assert (output_df.query('label_a == 3')[['za', 'zb']].values[0] == 0).all()
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (6,5)).all() # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == 2).all()

    assert (output_df.query('label_a == 6')[['za', 'zb']].values[0] == 0).all()
    assert (output_df.query('label_a == 6')[['ya', 'yb']].values[0] == (1,3)).all() # not 'forward'
    assert (output_df.query('label_a == 6')[['xa', 'xb']].values[0] == 6).all()

    return output_df


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

                      [_,1,1,2, 8,_,7,7], # 4
                      [_,_,_,_, _,_,_,_], # 5
                      [_,4,4,4, _,_,_,_], # 6
                      [_,3,3,3, _,5,9,_]]]# 7
    #                  0 1 2 3  4 5 6 7
    
    lowres_volume = np.zeros((4,8,8), np.uint64)
    lowres_volume[:] = volume_layout

    volume = upsample(lowres_volume, 16)
    assert volume.shape == (64,128,128)
 
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
        
          geometry:
            message-block-shape: [128,64,64]
 
        findadjacencies:
          output-table: output.csv
          find-closest-using-scale: 0
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
    config["findadjacencies"]["subset-labels"] = [1,2,3,4, 6,7,8]
    _impl_test_findadjacencies_from_dvid_sparse(template_dir, config)


def test_findadjacencies_from_dvid_sparse_edges(setup_dvid_segmentation_input):
    template_dir, config, _volume, _dvid_address, _repo_uuid = setup_dvid_segmentation_input

    subset_edges = pd.DataFrame([[1,2], [4,3], [7,6], [2,8]], columns=['label_a', 'label_b'])
    subset_edges.to_csv(f'{template_dir}/subset-edges.csv', index=False, header=True)
    
    # Overwrite config with updated settings.
    config = copy.deepcopy(config)
    config["findadjacencies"]["subset-edges"] = 'subset-edges.csv'

    execution_dir, workflow = _impl_test_findadjacencies_from_dvid_sparse(template_dir, config)

    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["findadjacencies"]["output-table"]}')

    # More checks
    label_pairs = output_df[['label_a', 'label_b']].values
    label_pairs = list(map(tuple, label_pairs))


def test_findadjacencies_from_dvid_sparse_groups(setup_dvid_segmentation_input):
    template_dir, config, _volume, _dvid_address, _repo_uuid = setup_dvid_segmentation_input

    # Overwrite config with updated settings.
    config = copy.deepcopy(config)
    config["findadjacencies"]["subset-label-groups"] = [[1,2,6,7,8], [1,2], [2,8], [3,4,5,9], [2,7], [8,6]]

    execution_dir, workflow = _impl_test_findadjacencies_from_dvid_sparse(template_dir, config)

    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["findadjacencies"]["output-table"]}')
    output_df = output_df.set_index(['label_a', 'label_b'])

    # More checks
    assert (1,2) in output_df.index
    assert (2,7) in output_df.index
    assert (6,8) in output_df.index
    assert (output_df.loc[(1,6), 'distance'] > 1).all()
    assert (output_df.loc[(1,7), 'distance'] > 1).all()
    assert (5,7) not in output_df.index
    assert (3,8) not in output_df.index


def _impl_test_findadjacencies_from_dvid_sparse(template_dir, config):
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    execution_dir, workflow = launch_flow(template_dir, 1)
    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["findadjacencies"]["output-table"]}')

    label_pairs = output_df[['label_a', 'label_b']].values
    assert 0 not in label_pairs.flat

    print()
    print(output_df)
    print()

    assert output_df[['label_a', 'label_b', 'group']].duplicated().sum() == 0

    label_pairs = list(map(tuple, label_pairs))
    assert (1,2) in label_pairs
    assert (3,4) in label_pairs
    assert (6,7) in label_pairs
    assert (2,8) in label_pairs
    
    assert (output_df.query('label_a == 3')[['za', 'zb']].values[0] == 31).all()
    assert (output_df.query('label_a == 3')[['ya', 'yb']].values[0] == (7*16, 7*16-1)).all() # not 'forward'
    assert (output_df.query('label_a == 3')[['xa', 'xb']].values[0] == 2.5*16-1).all()

    # The Z and X locations here are a little hard to test, since several voxels are tied.
    #assert (output_df.query('label_a == 6')[['za', 'zb']].values[0] == 31).all()
    assert (output_df.query('label_a == 6')[['ya', 'yb']].values[0] == (2*16-1, 3*16)).all() # not 'forward'
    assert (output_df.query('label_a == 6')[['xa', 'xb']].values[0] == 6*16).all()
    
    return execution_dir, workflow
    

def test_findadjacencies_different_dvid_blocks_sparse_labels(setup_dvid_segmentation_input):
    """
    There was a bug that caused a brick to be excluded from the computation if the objects
    of interest were in different DVID blocks, even though they were in the same brick.
    This checks that case.
    """
    template_dir, config, _volume, _dvid_address, _repo_uuid = setup_dvid_segmentation_input
    config = copy.deepcopy(config)

    # objects 1 and 6 are in different dvid blocks, but the same brick.
    config["findadjacencies"]["subset-labels"] = [1,6]
    _impl_findadjacencies_different_dvid_blocks_sparse_edges(template_dir, config)


def test_findadjacencies_different_dvid_blocks_sparse_edges(setup_dvid_segmentation_input):
    """
    There was a bug that caused a brick to be excluded from the computation if the objects
    of interest were in different DVID blocks, even though they were in the same brick.
    This checks that case.
    """
    template_dir, config, _volume, _dvid_address, _repo_uuid = setup_dvid_segmentation_input

    # objects 1 and 6 are in different dvid blocks, but the same brick.
    subset_edges = pd.DataFrame([[1,6]], columns=['label_a', 'label_b'])
    subset_edges.to_csv(f'{template_dir}/subset-edges.csv', index=False, header=True)

    # Overwrite config with updated settings.
    config = copy.deepcopy(config)
    config["findadjacencies"]["subset-edges"] = 'subset-edges.csv'
    _impl_findadjacencies_different_dvid_blocks_sparse_edges(template_dir, config)


def _impl_findadjacencies_different_dvid_blocks_sparse_edges(template_dir, config):
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    execution_dir, workflow = launch_flow(template_dir, 1)
    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["findadjacencies"]["output-table"]}')

    label_pairs = output_df[['label_a', 'label_b']].values
    assert 0 not in label_pairs.flat

    label_pairs = list(map(tuple, label_pairs))
    assert (1,6) in label_pairs


def test_append_group_col():
    edges = [[1, 2, 'a'], # duplicated in the output (1 and 2 both belong to the same two groups)
             [1, 2, 'b'], # duplicated in the output (1 and 2 both belong to the same two groups)
             [1, 3, 'c'],
             [1, 4, 'd'], # omitted from output (no groups in common)
             [2, 3, 'e'], # Both [2,3] edges must appear in the output, with the same group but preserved 'info' columns.
             [2, 3, 'f'], 
             [3, 4, 'g']] # omitted from output (no groups in common)

    groups = [[1, 10], [2, 10],
              [1, 20], [2, 20], [3, 20],
              [1, 30],
              [4, 40] ]

    expected = [[1, 2, 'a', 10],
                [1, 2, 'a', 20],
                [1, 2, 'b', 10],
                [1, 2, 'b', 20],
                [1, 3, 'c', 20],
                [2, 3, 'e', 20],
                [2, 3, 'f', 20]]

    edges_df = pd.DataFrame(edges, columns=['label_a', 'label_b', 'info'])
    groups_df = pd.DataFrame(groups, columns=['label', 'group'])

    expected_df = (pd.DataFrame(expected, columns=['label_a', 'label_b', 'info', 'group'])
                     .sort_values(['label_a', 'label_b', 'info', 'group'])
                     .reset_index(drop=True))[['group', 'label_a', 'label_b', 'info']]

    result_df = (append_group_col(edges_df, groups_df)
                    .sort_values(['label_a', 'label_b', 'info', 'group'])
                    .reset_index(drop=True))

    print("")
    print(expected_df)
    print("")
    print(result_df)
    assert (result_df == expected_df).all().all()


if __name__ == "__main__":
    #from neuclease import configure_default_logging
    #configure_default_logging()
    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_findadjacencies']
    args += ['-x']
    # #args += ['-k findadjacencies_from_dvid_sparse_groups']
    # args += ['-k findadjacencies_from_dvid_sparse_labels'
    #          ' or findadjacencies_from_dvid_sparse_edges'
    #          ' or findadjacencies_from_dvid_sparse_groups'
    # #         ' or findadjacencies_different_dvid_blocks_sparse_labels'
    # #         ' or findadjacencies_different_dvid_blocks_sparse_edges'
    # #         ' or findadjacencies_different_dvid_blocks_sparse_labels'
    #          ]
    pytest.main(args)
