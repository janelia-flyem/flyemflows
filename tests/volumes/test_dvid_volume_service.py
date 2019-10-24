import textwrap
from io import StringIO

import pytest
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from neuclease.util import extract_subvol, overwrite_subvol
from neuclease.dvid import fetch_repo_instances, fetch_instance_info

from flyemflows.util import downsample
from flyemflows.volumes import VolumeService


def test_dvid_volume_service_grayscale(setup_dvid_repo, disable_auto_retry):
    server, uuid = setup_dvid_repo
    instance_name = 'test-dvs-grayscale'

    volume = np.random.randint(100, size=(256, 192, 128), dtype=np.uint8)
    max_scale = 2
    voxel_dimensions = [4.0, 4.0, 32.0]

    config_text = textwrap.dedent(f"""\
        dvid:
          server: {server}
          uuid: {uuid}
          grayscale-name: {instance_name}
          
          create-if-necessary: true
          creation-settings:
            max-scale: {max_scale}
            voxel-size: {voxel_dimensions}
       
        geometry:
          bounding-box: [[0,0,0], {list(volume.shape[::-1])}]
    """)

    yaml = YAML()
    with StringIO(config_text) as f:
        volume_config = yaml.load(f)

    assert instance_name not in fetch_repo_instances(server, uuid)

    service = VolumeService.create_from_config( volume_config )

    repo_instances = fetch_repo_instances(server, uuid)
    
    info = fetch_instance_info(server, uuid, instance_name)
    assert info["Extended"]["VoxelSize"] == voxel_dimensions
    
    scaled_volumes = {}
    for scale in range(max_scale+1):
        if scale == 0:
            assert instance_name in repo_instances
            assert repo_instances[instance_name] == 'uint8blk'
        else:
            assert f"{instance_name}_{scale}" in repo_instances
            assert repo_instances[f"{instance_name}_{scale}"] == 'uint8blk'

        vol = downsample(volume, 2**scale, 'label') # label downsampling is easier to test with
        aligned_shape = (np.ceil(np.array(vol.shape) / 64) * 64).astype(int)
        aligned_vol = np.zeros(aligned_shape, np.uint8)
        overwrite_subvol(aligned_vol, [(0,0,0), aligned_shape], aligned_vol)
        service.write_subvolume(aligned_vol, (0,0,0), scale)
        scaled_volumes[scale] = aligned_vol
    
    box = np.array([[40, 80, 40], [240, 160, 100]])
    for scale in range(max_scale+1):
        scaled_box = box // 2**scale
        vol = service.get_subvolume(scaled_box, scale)
        assert (vol == extract_subvol(scaled_volumes[scale], scaled_box)).all()


def test_dvid_volume_service_labelmap(setup_dvid_repo, random_segmentation, disable_auto_retry):
    server, uuid = setup_dvid_repo
    instance_name = 'test-dvs-labelmap'

    volume = random_segmentation[:256, :192, :128]
    max_scale = 2
    voxel_dimensions = [4.0, 4.0, 32.0]

    config_text = textwrap.dedent(f"""\
        dvid:
          server: {server}
          uuid: {uuid}
          segmentation-name: {instance_name}
          supervoxels: true
          
          create-if-necessary: true
          creation-settings:
            max-scale: {max_scale}
            voxel-size: {voxel_dimensions}
       
        geometry:
          bounding-box: [[0,0,0], {list(volume.shape[::-1])}]
          message-block-shape: [64,64,64]
    """)

    yaml = YAML()
    with StringIO(config_text) as f:
        volume_config = yaml.load(f)

    assert instance_name not in fetch_repo_instances(server, uuid)

    service = VolumeService.create_from_config( volume_config )

    repo_instances = fetch_repo_instances(server, uuid)

    assert instance_name in repo_instances
    assert repo_instances[instance_name] == 'labelmap'

    info = fetch_instance_info(server, uuid, instance_name)
    assert info["Extended"]["VoxelSize"] == voxel_dimensions
    
    scaled_volumes = {}
    for scale in range(max_scale+1):
        vol = downsample(volume, 2**scale, 'label')
        aligned_shape = (np.ceil(np.array(vol.shape) / 64) * 64).astype(int)
        aligned_vol = np.zeros(aligned_shape, np.uint64)
        overwrite_subvol(aligned_vol, [(0,0,0), vol.shape], vol)
        
        service.write_subvolume(aligned_vol, (0,0,0), scale)
        scaled_volumes[scale] = aligned_vol
    
    box = np.array([[40, 80, 40], [240, 160, 100]])
    for scale in range(max_scale+1):
        scaled_box = box // 2**scale
        vol = service.get_subvolume(scaled_box, scale)
        assert (vol == extract_subvol(scaled_volumes[scale], scaled_box)).all()

    #
    # Check sparse coords function
    #
    labels = list({*pd.unique(volume.reshape(-1))} - {0})
    brick_coords_df = service.sparse_brick_coords_for_labels(labels)
    
    assert brick_coords_df.columns.tolist() == ['z','y','x','label']
    assert set(brick_coords_df['label'].values) == set(labels), \
        "Some labels were missing from the sparse brick coords!"

    def ndi(shape):
        return np.indices(shape).reshape(len(shape), -1).transpose()

    expected_df = pd.DataFrame(ndi(volume.shape), columns=[*'zyx'])
    
    expected_df['label'] = volume.reshape(-1)
    expected_df['z'] //= 64
    expected_df['y'] //= 64
    expected_df['x'] //= 64
    expected_df = expected_df.drop_duplicates()
    expected_df['z'] *= 64
    expected_df['y'] *= 64
    expected_df['x'] *= 64

    expected_df = expected_df.query('label != 0')

    expected_df.sort_values(['z', 'y', 'x', 'label'], inplace=True)
    brick_coords_df.sort_values(['z', 'y', 'x', 'label'], inplace=True)
    
    expected_df.reset_index(drop=True, inplace=True)
    brick_coords_df.reset_index(drop=True, inplace=True)
    
    assert expected_df.shape == brick_coords_df.shape
    assert (brick_coords_df == expected_df).all().all()

if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', 'tests.volumes.test_dvid_volume_service']
    #args += ['-k', 'dvid_volume_service_labelmap']
    pytest.main(args)
