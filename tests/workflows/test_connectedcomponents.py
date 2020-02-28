import os
import tempfile
import textwrap
from io import StringIO

import zarr
import h5py
import pytest
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from neuclease.util import ndrange

from flyemflows.util import upsample
from flyemflows.bin.launchflow import launch_flow
from neuclease.util import extract_subvol
from neuclease.dvid import create_labelmap_instance, post_labelmap_voxels, fetch_labelmap_voxels

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')


@pytest.fixture
def setup_connectedcomponents_hdf5_zarr():
    template_dir = tempfile.mkdtemp(suffix="connectedcomponents-template")
    
    _ = 0
    vol = [[_,_,_,_ ,_,_,_,_],
           [_,_,_,_ ,_,4,_,_],
           [_,1,1,1 ,_,_,1,1],
           [_,1,_,_ ,_,_,_,_],

           [_,1,_,_ ,_,_,_,_],
           [_,1,_,2 ,2,2,2,_],
           [_,_,_,_ ,_,_,_,_],
           [_,3,_,_ ,_,3,_,1]]
    
    # Create volume
    vol = np.array([vol], np.uint64)
    volume_path = f"{template_dir}/volume.h5"
    with h5py.File(volume_path, 'w') as f:
        f['volume'] = vol
    
    config = {
        "workflow-name": "connectedcomponents",
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
        "output": {
            "zarr": {
                "path": 'cc-vol.zarr',
                "dataset": "volume",
                "writable": True,
                "create-if-necessary": True,
                "creation-settings": {
                    "dtype": "uint64",
                    "chunk-shape": [4,4,1]
                }
            },
            "geometry": {
                "message-block-shape": [4,4,1],
                "block-width": 4
            },
        },
        
        "connectedcomponents": {
            "halo": 1
        }
    }

    yaml = YAML()
    yaml.default_flow_style = False
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    return template_dir, config, vol


def test_connectedcomponents(setup_connectedcomponents_hdf5_zarr, disable_auto_retry):
    template_dir, _config, input_vol = setup_connectedcomponents_hdf5_zarr

    execution_dir, workflow = launch_flow(template_dir, 1)
    final_config = workflow.config

    output_path = final_config["output"]["zarr"]["path"]
    dset_name = final_config["output"]["zarr"]["dataset"]

    store = zarr.NestedDirectoryStore(output_path)
    f = zarr.open(store=store, mode='r')
    output_vol = f[dset_name][:]
    assert output_vol.shape == input_vol.shape

    final_labels = pd.unique(output_vol.reshape(-1))
    
    # Never change label 0
    assert 0 in final_labels
    assert ((input_vol == 0) == (output_vol == 0)).all()
    
    # Single-component objects
    assert 2 in final_labels
    assert 4 in final_labels

    assert ((input_vol == 2) == (output_vol == 2)).all()
    assert ((input_vol == 4) == (output_vol == 4)).all()

    # Split objects
    assert 1 not in final_labels
    assert 3 not in final_labels
    
    for corner in map(np.array, ndrange((0,0,0), (1,8,8), (1,4,4))):
        box = (corner, corner + (1,4,4))
        input_block = extract_subvol(input_vol, box)
        output_block = extract_subvol(output_vol, box)
        
        for orig_label in [1,3]:
            if orig_label in input_block:
                positions = (input_block == orig_label)

                assert (input_block[positions] != output_block[positions]).all(), \
                    f"original label {orig_label} was not split!"

                assert (output_block[positions] > input_vol.max()).all(), \
                    f"original label {orig_label} was not split!"
                
                # This block-based assertion is not generally true for all possible input,
                # but our test data blocks are set up so that this is a valid check.
                # (No block happens to contain more than one final CC that came from the same original label.)
                assert (output_block[positions] == output_block[positions][0]).all(), \
                    f"original label {orig_label} ended up over-segmentated"

    # Check CSV output
    df = pd.read_csv(f'{execution_dir}/relabeled-objects.csv')
    
    assert len(df.query('orig_label == 0')) == 0
    assert len(df.query('orig_label == 1')) == 3
    assert len(df.query('orig_label == 2')) == 0
    assert len(df.query('orig_label == 3')) == 2
    assert len(df.query('orig_label == 4')) == 0

    assert not df['final_label'].duplicated().any()
    assert (df['final_label'] > input_vol.max()).all()


def test_connectedcomponents_subset_labels(setup_connectedcomponents_hdf5_zarr, disable_auto_retry):
    template_dir, config, input_vol = setup_connectedcomponents_hdf5_zarr

    config["connectedcomponents"]["subset-labels"] = [1,2,4] # Not 3

    # Overwrite config
    yaml = YAML()
    yaml.default_flow_style = False
    
    # re-dump config in case it's been changed by a specific test
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    execution_dir, workflow = launch_flow(template_dir, 1)
    final_config = workflow.config

    output_path = final_config["output"]["zarr"]["path"]
    dset_name = final_config["output"]["zarr"]["dataset"]
    
    store = zarr.NestedDirectoryStore(output_path)
    f = zarr.open(store=store, mode='r')
    output_vol = f[dset_name][:]
    assert output_vol.shape == input_vol.shape

    final_labels = pd.unique(output_vol.reshape(-1))
    
    # Never change label 0
    assert 0 in final_labels
    assert ((input_vol == 0) == (output_vol == 0)).all()
    
    # Single-component objects
    assert 2 in final_labels
    assert 4 in final_labels

    assert ((input_vol == 2) == (output_vol == 2)).all()
    assert ((input_vol == 4) == (output_vol == 4)).all()

    # Omitted from analysis; left unsplit
    assert 3 in final_labels
    assert ((input_vol == 3) == (output_vol == 3)).all()

    # Split objects
    assert 1 not in final_labels
    
    for corner in map(np.array, ndrange((0,0,0), (1,8,8), (1,4,4))):
        box = (corner, corner + (1,4,4))
        input_block = extract_subvol(input_vol, box)
        output_block = extract_subvol(output_vol, box)
        
        for orig_label in [1]:
            if orig_label in input_block:
                positions = (input_block == orig_label)

                assert (input_block[positions] != output_block[positions]).all(), \
                    f"original label {orig_label} was not split!"

                assert (output_block[positions] > input_vol.max()).all(), \
                    f"original label {orig_label} was not split!"
                
                # This block-based assertion is not generally true for all possible input,
                # but our test data blocks are set up so that this is a valid check.
                # (No block happens to contain more than one final CC that came from the same original label.)
                assert (output_block[positions] == output_block[positions][0]).all(), \
                    f"original label {orig_label} ended up over-segmentated"

    # Check CSV output
    df = pd.read_csv(f'{execution_dir}/relabeled-objects.csv')
    
    assert len(df.query('orig_label == 0')) == 0
    assert len(df.query('orig_label == 1')) == 3
    assert len(df.query('orig_label == 2')) == 0
    assert len(df.query('orig_label == 3')) == 0 # 3 was not touched.
    assert len(df.query('orig_label == 4')) == 0

    assert not df['final_label'].duplicated().any()
    assert (df['final_label'] > input_vol.max()).all()


@pytest.fixture
def setup_connectedcomponents_dvid(setup_dvid_repo):
    dvid_address, repo_uuid = setup_dvid_repo
    
    _ = 0
    volume_layout = [[_,_,_,_ ,_,_,_,_],
                     [_,_,_,_ ,_,4,_,_],
                     [_,1,1,1 ,_,_,1,1],
                     [_,1,_,_ ,_,_,_,_],

                     [_,1,_,_ ,_,_,_,_],
                     [_,1,_,2 ,2,2,2,_],
                     [_,_,_,_ ,_,_,_,_],
                     [_,3,_,_ ,_,3,_,1]]
    
    lowres_volume = np.zeros((4,8,8), np.uint64)
    lowres_volume[:] = volume_layout

    volume = upsample(lowres_volume, 16)
    assert volume.shape == (64,128,128)

    input_segmentation_name = 'cc-input'
    output_segmentation_name = 'cc-output'
 
    create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name)
    post_labelmap_voxels(dvid_address, repo_uuid, input_segmentation_name, (0,0,0), volume)

    # Post data to the output -- don't leave it empty,
    # or we run into 'maxlabel' issues related to dvid issue #284
    # https://github.com/janelia-flyem/dvid/issues/284
    create_labelmap_instance(dvid_address, repo_uuid, output_segmentation_name)
    post_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, (0,0,0), volume)

    config_text = textwrap.dedent(f"""\
        workflow-name: connectedcomponents
        cluster-type: {CLUSTER_TYPE}
         
        input:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {input_segmentation_name}
            supervoxels: true
           
          geometry:
            message-block-shape: [64,64,64]
            bounding-box: [[0,0,0], [128,128,64]]
 
        output:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {output_segmentation_name}
            supervoxels: true
            disable-indexing: true
            create-if-necessary: true
           
          geometry: {{}} # Auto-set from input
 
        connectedcomponents:
          halo: 1
          subset-labels: [1,2,4] # Not 3
          compute-block-statistics: true
    """)
 
    template_dir = tempfile.mkdtemp(suffix="connectedcomponents-template")

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)

    yaml = YAML()
    yaml.default_flow_style = False
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    return template_dir, config, volume, dvid_address, repo_uuid, output_segmentation_name


def test_connectedcomponents_dvid_subset_labels(setup_connectedcomponents_dvid, disable_auto_retry):
    template_dir, _config, input_vol, dvid_address, repo_uuid, output_segmentation_name = setup_connectedcomponents_dvid

    execution_dir, workflow = launch_flow(template_dir, 1)
    _final_config = workflow.config

    output_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, [(0,0,0), input_vol.shape], supervoxels=True)
    assert output_vol.shape == input_vol.shape

    final_labels = pd.unique(output_vol.reshape(-1))
    
    # Never change label 0
    assert 0 in final_labels
    assert ((input_vol == 0) == (output_vol == 0)).all()
    
    # Single-component objects
    assert 2 in final_labels
    assert 4 in final_labels

    assert ((input_vol == 2) == (output_vol == 2)).all()
    assert ((input_vol == 4) == (output_vol == 4)).all()

    # Omitted from analysis; left unsplit
    assert 3 in final_labels
    assert ((input_vol == 3) == (output_vol == 3)).all()

    # Split objects
    assert 1 not in final_labels
    
    for corner in map(np.array, ndrange((0,0,0), (1,8,8), (1,4,4))):
        box = (corner, corner + 4)
        input_block = extract_subvol(input_vol, box)
        output_block = extract_subvol(output_vol, box)
        
        for orig_label in [1]:
            if orig_label in input_block:
                positions = (input_block == orig_label)

                assert (input_block[positions] != output_block[positions]).all(), \
                    f"original label {orig_label} was not split!"

                assert (output_block[positions] > input_vol.max()).all(), \
                    f"original label {orig_label} was not split!"
                
                # This block-based assertion is not generally true for all possible input,
                # but our test data blocks are set up so that this is a valid check.
                # (No block happens to contain more than one final CC that came from the same original label.)
                assert (output_block[positions] == output_block[positions][0]).all(), \
                    f"original label {orig_label} ended up over-segmentated"

    #
    # Check CSV output
    #
    df = pd.read_csv(f'{execution_dir}/relabeled-objects.csv')
    
    assert len(df.query('orig_label == 0')) == 0
    assert len(df.query('orig_label == 1')) == 3
    assert len(df.query('orig_label == 2')) == 0
    assert len(df.query('orig_label == 3')) == 0 # 3 was not touched.
    assert len(df.query('orig_label == 4')) == 0

    assert not df['final_label'].duplicated().any()
    assert (df['final_label'] > input_vol.max()).all()

    #
    # Check block stats
    #
    with h5py.File(f'{execution_dir}/block-statistics.h5', 'r') as f:
        stats_df = pd.DataFrame(f['stats'][:])
    
    for row in stats_df.itertuples():
        corner = np.array((row.z, row.y, row.x))
        block_box = np.array([corner, corner+64])
        block = extract_subvol(output_vol, block_box)
        assert (block == row.segment_id).sum() == row.count

if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")
    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_connectedcomponents']
    #args = ['-k', 'connectedcomponents'] + args
    pytest.main(args)
