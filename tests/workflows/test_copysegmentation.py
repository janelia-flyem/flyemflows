import os
import tempfile
import textwrap
from io import StringIO

import h5py
import numpy as np

from dvidutils import downsample_labels
from neuclease.util import box_to_slicing
from neuclease.dvid import create_labelmap_instance, post_labelarray_voxels, fetch_raw, fetch_labelarray_voxels

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchflow import launch_flow

TESTVOL_SHAPE = (256,256,256)

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')
#CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'synchronous')

@pytest.fixture
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

@pytest.fixture
def setup_dvid_segmentation_input(setup_dvid_repo, random_segmentation):
    dvid_address, repo_uuid = setup_dvid_repo
 
    input_segmentation_name = 'segmentation-input'
    output_segmentation_name = 'segmentation-output-from-dvid'
 
    create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name)
    post_labelarray_voxels(dvid_address, repo_uuid, input_segmentation_name, (0,0,0), random_segmentation)
    
    template_dir = tempfile.mkdtemp(suffix="copysegmentation-from-dvid-template")
 
    config_text = textwrap.dedent(f"""\
        workflow-name: copysegmentation
        cluster-type: {CLUSTER_TYPE}
         
        input:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {input_segmentation_name}
            supervoxels: true
           
          geometry:
            message-block-shape: [64,64,512]
            bounding-box: [[0,0,100], [256,200,256]]
 
        outputs:
          - dvid:
              server: {dvid_address}
              uuid: {repo_uuid}
              segmentation-name: {output_segmentation_name}
              supervoxels: true
              disable-indexing: true
           
            geometry: {{}} # Auto-set from input
 
        copysegmentation:
          pyramid-depth: 1
          slab-depth: 128
    """)
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)
 
    return template_dir, config, random_segmentation, dvid_address, repo_uuid, output_segmentation_name


@pytest.fixture
def setup_hdf5_segmentation_input(setup_dvid_repo, random_segmentation):
    dvid_address, repo_uuid = setup_dvid_repo
    template_dir = tempfile.mkdtemp(suffix="copysegmentation-from-hdf5-template")
    
    volume_path = f"{template_dir}/volume.h5"
    with h5py.File(volume_path, 'w') as f:
        f['volume'] = random_segmentation

    output_segmentation_name = 'segmentation-output-from-hdf5'
    
    config_text = textwrap.dedent(f"""\
        workflow-name: copysegmentation
        cluster-type: {CLUSTER_TYPE}
        
        input:
          hdf5:
            path: {volume_path}
            dataset: volume
          
          geometry:
            message-block-shape: [64,64,256]
            bounding-box: [[0,0,100], [256,200,256]]

        outputs:
          - dvid:
              server: {dvid_address}
              uuid: {repo_uuid}
              segmentation-name: {output_segmentation_name}
              supervoxels: true
              disable-indexing: true
                        
            geometry: {{}} # Auto-set from input
        
        copysegmentation:
          pyramid-depth: 1
          slab-depth: 128
          download-pre-downsampled: false
    """)

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)

    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)

    return template_dir, config, random_segmentation, dvid_address, repo_uuid, output_segmentation_name


def _run(setup, check_scale_0=True):
    template_dir, config, volume, dvid_address, repo_uuid, output_segmentation_name = setup

    yaml = YAML()
    yaml.default_flow_style = False
    
    # re-dump config in case it's been changed by a specific test
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    _execution_dir, workflow = launch_flow(template_dir, 1)
    final_config = workflow.config

    box_xyz = np.array( final_config['input']['geometry']['bounding-box'] )
    box_zyx = box_xyz[:,::-1]
    
    output_vol = fetch_raw(dvid_address, repo_uuid, output_segmentation_name, box_zyx, dtype=np.uint64)
    expected_vol = volume[box_to_slicing(*box_zyx)]
    
    if check_scale_0:
        assert (output_vol == expected_vol).all(), \
            "Written vol does not match expected"
    
    return box_zyx, expected_vol


def test_copysegmentation_from_dvid_to_dvid(setup_dvid_segmentation_input):
    _box_zyx, _expected_vol = _run(setup_dvid_segmentation_input)
   
   
def test_copysegmentation_from_hdf5_to_dvid(setup_hdf5_segmentation_input):
    _box_zyx, _expected_vol = _run(setup_hdf5_segmentation_input)
 
 
def test_copysegmentation_from_hdf5_to_dvid_multiscale(setup_hdf5_segmentation_input):
    template_dir, config, volume, dvid_address, repo_uuid, _ = setup_hdf5_segmentation_input
    
    # Modify the config from above to compute pyramid scales,
    # and choose a bounding box that is aligned with the bricks even at scale 2
    # (just for easier testing).
    box_zyx = [[0,0,0],[256,256,256]]
    config["input"]["geometry"]["bounding-box"] = box_zyx
    config["copysegmentation"]["pyramid-depth"] = 2

    # Change the segmentation name so it doesn't conflict with earlier tests 
    output_segmentation_name = 'segmentation-output-from-hdf5-multiscale'
    config["outputs"][0]["dvid"]["segmentation-name"] = output_segmentation_name

    yaml = YAML()
    yaml.default_flow_style = False
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    _execution_dir, _workflow = launch_flow(template_dir, 1)

    box_zyx = np.array(box_zyx)

    scale_0_vol = volume[box_to_slicing(*box_zyx)]
    scale_1_vol = downsample_labels(scale_0_vol, 2, True)
    scale_2_vol = downsample_labels(scale_1_vol, 2, True)
 
    output_0_vol = fetch_labelarray_voxels(dvid_address, repo_uuid, output_segmentation_name, box_zyx // 1, scale=0)
    output_1_vol = fetch_labelarray_voxels(dvid_address, repo_uuid, output_segmentation_name, box_zyx // 2, scale=1)
    output_2_vol = fetch_labelarray_voxels(dvid_address, repo_uuid, output_segmentation_name, box_zyx // 4, scale=2)

#     np.save('/tmp/expected-0.npy', scale_0_vol)
#     np.save('/tmp/expected-1.npy', scale_1_vol)
#     np.save('/tmp/expected-2.npy', scale_2_vol)
# 
#     np.save('/tmp/output-0.npy', output_0_vol)
#     np.save('/tmp/output-1.npy', output_1_vol)
#     np.save('/tmp/output-2.npy', output_2_vol)
# 
#     np.save('/tmp/diff-0.npy', (output_0_vol != scale_0_vol))
#     np.save('/tmp/diff-1.npy', (output_1_vol != scale_1_vol))
#     np.save('/tmp/diff-2.npy', (output_2_vol != scale_2_vol))

    assert (output_0_vol == scale_0_vol).all(), \
        "Scale 0: Written vol does not match expected"
    assert (output_1_vol == scale_1_vol).all(), \
        "Scale 1: Written vol does not match expected"
    assert (output_2_vol == scale_2_vol).all(), \
        "Scale 2: Written vol does not match expected"

if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")
    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_copysegmentation'])
