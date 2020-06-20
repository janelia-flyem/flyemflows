import os
import pickle
import tempfile
import textwrap
from io import StringIO

import pytest
from ruamel.yaml import YAML
from requests import HTTPError

import h5py
import numpy as np
import pandas as pd

from dvidutils import downsample_labels

from neuclease.util import box_to_slicing, extract_subvol, overwrite_subvol, box_intersection, mask_for_labels, apply_mask_for_labels, SparseBlockMask
from neuclease.dvid import fetch_repo_instances, create_labelmap_instance, post_labelmap_voxels, fetch_raw, fetch_labelmap_voxels

from flyemflows.bin.launchflow import launch_flow

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')

@pytest.fixture
def setup_dvid_segmentation_input(setup_dvid_repo, random_segmentation):
    dvid_address, repo_uuid = setup_dvid_repo
 
    input_segmentation_name = 'segmentation-input'
    output_segmentation_name = 'segmentation-output-from-dvid'
 
    try:
        create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name)
    except HTTPError as ex:
        if ex.response is not None and 'already exists' in ex.response.content.decode('utf-8'):
            pass
        
    post_labelmap_voxels(dvid_address, repo_uuid, input_segmentation_name, (0,0,0), random_segmentation)

    # Make sure the output is empty (if it exists)
    if output_segmentation_name in fetch_repo_instances(dvid_address, repo_uuid):
        z = np.zeros((256, 256, 256), np.uint64)
        post_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, (0,0,0), z, 0, True)

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
          
          adapters: {{}}
 
        output:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {output_segmentation_name}
            supervoxels: true
            disable-indexing: true
            create-if-necessary: true
           
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


@pytest.fixture(scope='module')
def write_hdf5_volume(random_segmentation):
    template_dir = tempfile.mkdtemp(suffix="copysegmentation-hdf5-volume")
    volume_path = f"{template_dir}/volume.h5"
    with h5py.File(volume_path, 'w') as f:
        f['volume'] = random_segmentation
    return volume_path, random_segmentation


@pytest.fixture
def setup_hdf5_segmentation_input(setup_dvid_repo, write_hdf5_volume):
    volume_path, random_segmentation = write_hdf5_volume
    dvid_address, repo_uuid = setup_dvid_repo
    template_dir = tempfile.mkdtemp(suffix="copysegmentation-from-hdf5-template")
    
    output_segmentation_name = 'segmentation-output-from-hdf5'

    # Make sure the output is empty (if it exists)
    if output_segmentation_name in fetch_repo_instances(dvid_address, repo_uuid):
        z = np.zeros((256, 256, 256), np.uint64)
        post_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, (0,0,0), z, 0, True)
    
    config_text = textwrap.dedent(f"""\
        workflow-name: copysegmentation
        cluster-type: {CLUSTER_TYPE}
        
        input:
          hdf5:
            path: {volume_path}
            dataset: volume
          
          geometry:
            message-block-shape: [64,64,256] # note: this is weird because normally we stripe in the X direction...
            bounding-box: [[0,0,100], [256,200,256]]

        output:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {output_segmentation_name}
            supervoxels: true
            disable-indexing: true
            create-if-necessary: true
                        
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


def _run_to_dvid(setup, check_scale_0=True):
    template_dir, config, volume, dvid_address, repo_uuid, output_segmentation_name = setup

    yaml = YAML()
    yaml.default_flow_style = False
    
    # re-dump config in case it's been changed by a specific test
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    _execution_dir, workflow = launch_flow(template_dir, 1)
    final_config = workflow.config

    input_box_xyz = np.array( final_config['input']['geometry']['bounding-box'] )
    input_box_zyx = input_box_xyz[:,::-1]
    
    expected_vol = extract_subvol(volume, input_box_zyx)
    
    output_box_xyz = np.array( final_config['output']['geometry']['bounding-box'] )
    output_box_zyx = output_box_xyz[:,::-1]
    output_vol = fetch_raw(dvid_address, repo_uuid, output_segmentation_name, output_box_zyx, dtype=np.uint64)

    np.save('/tmp/output_vol.npy', output_vol)
    np.save('/tmp/expected_vol.npy', expected_vol)

    if check_scale_0:
        assert (output_vol == expected_vol).all(), \
            "Written vol does not match expected"
    
    return input_box_zyx, expected_vol, output_vol


def test_copysegmentation_from_dvid_to_dvid(setup_dvid_segmentation_input, disable_auto_retry):
    _box_zyx, _expected_vol, _output_vol = _run_to_dvid(setup_dvid_segmentation_input)


def test_copysegmentation_from_dvid_to_dvid_with_labelmap(setup_dvid_segmentation_input, disable_auto_retry):
    template_dir, config, volume, dvid_address, repo_uuid, _output_segmentation_name = setup_dvid_segmentation_input

    # make sure we get a fresh output
    output_segmentation_name = 'copyseg-with-labelmap'
    config["output"]["dvid"]["segmentation-name"] = output_segmentation_name

    orig_labels = pd.unique(volume.reshape(-1))
    new_labels = orig_labels + 4000
    expected_vol = volume + 4000

    pd.DataFrame({'orig': orig_labels, 'new': new_labels}).to_csv(f"{template_dir}/labelmap.csv", header=True, index=False)
    config["input"]["adapters"]["apply-labelmap"] = {
        "file": "labelmap.csv",
        "file-type": "label-to-body"
    }

    setup = template_dir, config, expected_vol, dvid_address, repo_uuid, output_segmentation_name
    _box_zyx, _expected_vol, _output_vol = _run_to_dvid(setup)



def test_copysegmentation_from_dvid_to_dvid_input_mask(setup_dvid_segmentation_input, disable_auto_retry):
    template_dir, config, volume, dvid_address, repo_uuid, _output_segmentation_name = setup_dvid_segmentation_input
    
    # make sure we get a fresh output
    output_segmentation_name = 'copyseg-with-input-mask-from-dvid'
    config["output"]["dvid"]["segmentation-name"] = output_segmentation_name

    # Add an offset, which is added to both the input volume AND the mask labels
    offset = 2000
    config["copysegmentation"]["add-offset-to-ids"] = offset

    # Select some labels that don't extend throughout the whole volume
    selected_labels = pd.unique( volume[150, 64:128, 64:128].reshape(-1) )
    assert 0 not in selected_labels
    selected_coords = np.array(mask_for_labels(volume, selected_labels).nonzero()).transpose()
    selected_box = np.array([selected_coords.min(axis=0), 1+selected_coords.max(axis=0)])

    input_box = np.array(config["input"]["geometry"]["bounding-box"])[:,::-1]

    subvol_box = box_intersection(input_box, selected_box)
    selected_subvol = extract_subvol(volume, subvol_box).copy()
    selected_subvol = apply_mask_for_labels(selected_subvol, selected_labels)
    config["copysegmentation"]["input-mask-labels"] = selected_labels.tolist()
    
    selected_subvol = np.where(selected_subvol, selected_subvol+offset, 0)
    expected_vol = np.zeros(volume.shape, np.uint64)
    overwrite_subvol(expected_vol, subvol_box, selected_subvol) 

    setup = template_dir, config, expected_vol, dvid_address, repo_uuid, output_segmentation_name
    _box_zyx, _expected_vol, _output_vol = _run_to_dvid(setup)


# def test_copysegmentation_from_dvid_to_dvid_both_masks(setup_dvid_segmentation_input, disable_auto_retry):
# TODO


def test_copysegmentation_from_hdf5_to_dvid(setup_hdf5_segmentation_input, disable_auto_retry):
    _box_zyx, _expected_vol, _output_vol = _run_to_dvid(setup_hdf5_segmentation_input)
 

def test_copysegmentation_from_hdf5_to_dvid_custom_sbm(setup_hdf5_segmentation_input, disable_auto_retry):
    template_dir, config, volume, dvid_address, repo_uuid, output_segmentation_name = setup_hdf5_segmentation_input

    # Our bricks are long in Z, so use a mask that's aligned that way, too.
    mask = np.zeros(volume.shape, bool)
    mask[:, :, 64:128] = True
    mask[:, :, 192:256] = True

    sbm = SparseBlockMask(mask[::64, ::64, ::64], [(0,0,0), volume.shape], (64,64,64))
    with open(f"{template_dir}/sbm.pkl", 'wb') as f:
        pickle.dump(sbm, f)
    config["copysegmentation"]["sparse-block-mask"] = f"{template_dir}/sbm.pkl"

    setup = (template_dir, config, volume, dvid_address, repo_uuid, output_segmentation_name)
    box_zyx, expected_vol, output_vol = _run_to_dvid(setup, check_scale_0=False)

    expected_vol = expected_vol.copy()
    mask = mask[box_to_slicing(*box_zyx)]
    expected_vol[~mask] = 0
    assert (output_vol == expected_vol).all()

def test_copysegmentation_from_hdf5_to_dvid_input_mask(setup_hdf5_segmentation_input, disable_auto_retry):
    template_dir, config, volume, dvid_address, repo_uuid, _output_segmentation_name = setup_hdf5_segmentation_input
    
    # make sure we get a fresh output
    output_segmentation_name = 'copyseg-with-input-mask'
    config["output"]["dvid"]["segmentation-name"] = output_segmentation_name

    # Select only even IDs
    all_labels = pd.unique(volume.reshape(-1))
    even_labels = all_labels[all_labels % 2 == 0]
    config["copysegmentation"]["input-mask-labels"] = even_labels.tolist()
    
    # Add an offset, which is added to both the input volume AND the mask labels
    offset = 2000 
    config["copysegmentation"]["add-offset-to-ids"] = offset

    input_box = np.array(config["input"]["geometry"]["bounding-box"])[:,::-1]
    volume = np.where((volume % 2) == 0, volume+offset, 0)
    expected_vol = np.zeros_like(volume)
    overwrite_subvol(expected_vol, input_box, extract_subvol(volume, input_box))

    setup = template_dir, config, expected_vol, dvid_address, repo_uuid, output_segmentation_name
    _box_zyx, _expected_vol, _output_vol = _run_to_dvid(setup)



def test_copysegmentation_from_hdf5_to_dvid_output_mask(setup_hdf5_segmentation_input, disable_auto_retry):
    template_dir, config, input_volume, dvid_address, repo_uuid, _output_segmentation_name = setup_hdf5_segmentation_input

    # make sure we get a fresh output
    output_segmentation_name = 'copyseg-with-output-mask'
    config["output"]["dvid"]["segmentation-name"] = output_segmentation_name

    output_volume = np.zeros((256,256,256), np.uint64)
    mask = np.zeros((256,256,256), dtype=bool)
    
    masked_labels = [5, 10, 15, 20]

    # Start with an output that is striped (but along on block boundaries)
    for label, (z_start, z_stop) in enumerate(zip(range(0,250, 10), range(10, 260, 10))):
        output_volume[z_start:z_stop] = label
        if label in masked_labels:
            mask[z_start:z_stop] = True

    # We expect the output to remain unchanged except in the masked voxels.
    expected_vol = np.where(mask, input_volume, output_volume)

    # make sure we get a fresh output
    output_segmentation_name = 'copyseg-with-output-mask'
    config["output"]["dvid"]["segmentation-name"] = output_segmentation_name
    config["copysegmentation"]["output-mask-labels"] = masked_labels

    max_scale = config["copysegmentation"]["pyramid-depth"]
    create_labelmap_instance(dvid_address, repo_uuid, output_segmentation_name, max_scale=max_scale)
    post_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, (0,0,0), output_volume)

    setup = template_dir, config, expected_vol, dvid_address, repo_uuid, output_segmentation_name
    _box_zyx, _expected_vol, _output_vol = _run_to_dvid(setup)


def test_copysegmentation_from_hdf5_to_dvid_multiscale(setup_hdf5_segmentation_input, disable_auto_retry):
    template_dir, config, volume, dvid_address, repo_uuid, _ = setup_hdf5_segmentation_input
    
    # Modify the config from above to compute pyramid scales,
    # and choose a bounding box that is aligned with the bricks even at scale 2
    # (just for easier testing).
    box_zyx = [[0,0,0],[256,256,256]]
    config["input"]["geometry"]["bounding-box"] = box_zyx
    config["copysegmentation"]["pyramid-depth"] = 2

    # Change the segmentation name so it doesn't conflict with earlier tests 
    output_segmentation_name = 'segmentation-output-from-hdf5-multiscale'
    config["output"]["dvid"]["segmentation-name"] = output_segmentation_name

    yaml = YAML()
    yaml.default_flow_style = False
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    _execution_dir, _workflow = launch_flow(template_dir, 1)

    box_zyx = np.array(box_zyx)

    scale_0_vol = volume[box_to_slicing(*box_zyx)]
    scale_1_vol = downsample_labels(scale_0_vol, 2, True)
    scale_2_vol = downsample_labels(scale_1_vol, 2, True)
 
    output_0_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, box_zyx // 1, scale=0)
    output_1_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, box_zyx // 2, scale=1)
    output_2_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, box_zyx // 4, scale=2)

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


@pytest.mark.skipif(not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', None), reason="Skipping Brainmaps test")
def test_copysegmentation_from_brainmaps_to_dvid(setup_dvid_repo):
    """
    Fetch a tiny subvolume from a Brainmaps source.
    To run this test, you must have valid application credentials loaded in your bash environment,
    
    For example:
        
        export GOOGLE_APPLICATION_CREDENTIALS=/Users/bergs/dvid-em-28a78d822e11.json
        PYTHONPATH=. pytest -s --tb=native --pyargs tests.workflows.test_copysegmentation -k copysegmentation_from_brainmaps_to_dvid
    """
    dvid_address, repo_uuid = setup_dvid_repo
    output_segmentation_name = 'segmentation-output-from-brainmaps'

    box_start = np.array([8000, 23296, 12800])
    box_xyz = np.array([box_start, box_start + 256])
    box_zyx = box_xyz[:,::-1]

    config_text = textwrap.dedent(f"""\
        workflow-name: copysegmentation
        cluster-type: {CLUSTER_TYPE}
         
        input:
          brainmaps:
            project: '274750196357'
            dataset: hemibrain
            volume-id: base20180227_8nm_watershed_fixed
            change-stack-id: ''

            # Uh-oh, apparently this change stack is no longer available in BrainMaps??
            #change-stack-id: ffn_agglo_20180312_32_16_8_freeze10

          geometry:
            bounding-box: {box_xyz.tolist()}
            message-block-shape: [6400, 64, 64]
            block-width: 64
            available-scales: [0,1,2]

        output:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {output_segmentation_name}
            supervoxels: true
            disable-indexing: true
            create-if-necessary: true
           
          geometry: {{}} # Auto-set from input
 
        copysegmentation:
          pyramid-depth: 2
          slab-depth: 128
          download-pre-downsampled: true
    """)
 
    template_dir = tempfile.mkdtemp(suffix="copysegmentation-from-brainmaps")
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)
 
    _execution_dir, _workflow = launch_flow(template_dir, 1)

    # Fetch the data via a simpler method, and verify that it matches what we stored in DVID.
    from flyemflows.volumes.brainmaps_volume import BrainMapsVolume
    bmv = BrainMapsVolume.from_flyem_source_info(config['input']['brainmaps'])
    
    
    for scale in (0,1,2):
        expected_vol = bmv.get_subvolume(box_zyx // 2**scale, scale=scale)
    
        assert expected_vol.any(), \
            f"Something is wrong with this test: The brainmaps volume at scale {scale} is all zeros!"
        
        output_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, box_zyx // 2**scale, scale=scale)
        assert (output_vol == expected_vol).all()


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")
    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_copysegmentation']
    args += ['-x']
    #args += ['-k', 'copysegmentation_from_hdf5_to_dvid_custom_sbm or copysegmentation_from_hdf5_to_dvid_input_mask']
    pytest.main(args)
