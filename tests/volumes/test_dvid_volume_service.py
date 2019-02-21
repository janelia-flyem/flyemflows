import textwrap
from io import StringIO

import pytest
import numpy as np
from ruamel.yaml import YAML

from neuclease.util import extract_subvol, overwrite_subvol
from neuclease.dvid import fetch_repo_instances

from flyemflows.util import downsample
from flyemflows.volumes import VolumeService


def test_dvid_volume_service_grayscale(setup_dvid_repo, disable_auto_retry):
    server, uuid = setup_dvid_repo
    instance_name = 'test-dvs-grayscale'

    volume = np.random.randint(100, size=(256, 192, 128), dtype=np.uint8)
    max_scale = 2

    config_text = textwrap.dedent(f"""\
        dvid:
          server: {server}
          uuid: {uuid}
          grayscale-name: {instance_name}
          
          create-if-necessary: true
          creation-settings:
            max-scale: {max_scale}
       
        geometry:
          bounding-box: [[0,0,0], {list(volume.shape[::-1])}]
    """)

    yaml = YAML()
    with StringIO(config_text) as f:
        volume_config = yaml.load(f)

    assert instance_name not in fetch_repo_instances(server, uuid)

    service = VolumeService.create_from_config( volume_config )

    repo_instances = fetch_repo_instances(server, uuid)
    
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


def test_dvid_volume_service_labelmap(setup_dvid_repo, disable_auto_retry):
    server, uuid = setup_dvid_repo
    instance_name = 'test-dvs-labelmap'

    volume = np.random.randint(100, size=(256, 192, 128), dtype=np.uint64)
    max_scale = 2

    config_text = textwrap.dedent(f"""\
        dvid:
          server: {server}
          uuid: {uuid}
          segmentation-name: {instance_name}
          supervoxels: true
          
          create-if-necessary: true
          creation-settings:
            max-scale: {max_scale}
       
        geometry:
          bounding-box: [[0,0,0], {list(volume.shape[::-1])}]
    """)

    yaml = YAML()
    with StringIO(config_text) as f:
        volume_config = yaml.load(f)

    assert instance_name not in fetch_repo_instances(server, uuid)

    service = VolumeService.create_from_config( volume_config )

    repo_instances = fetch_repo_instances(server, uuid)

    assert instance_name in repo_instances
    assert repo_instances[instance_name] == 'labelmap'
    
    scaled_volumes = {}
    for scale in range(max_scale+1):

        vol = downsample(volume, 2**scale, 'label')
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


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.volumes.test_dvid_volume_service'])
