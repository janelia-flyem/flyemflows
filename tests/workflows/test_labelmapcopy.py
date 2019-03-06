import os
import tempfile
import textwrap
from io import StringIO

import pytest
from ruamel.yaml import YAML
from requests import HTTPError

import numpy as np

from neuclease.dvid import create_labelmap_instance, post_labelmap_voxels, fetch_labelmap_voxels

from flyemflows.util import downsample
from flyemflows.bin.launchflow import launch_flow

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')


@pytest.fixture
def setup_dvid_segmentation_input(setup_dvid_repo, random_segmentation):
    dvid_address, repo_uuid = setup_dvid_repo
 
    input_segmentation_name = 'labelmapcopy-segmentation-input'
    output_segmentation_name = 'labelmapcopy-segmentation-output'
 
    max_scale = 2
    try:
        create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name, max_scale=max_scale)
    except HTTPError as ex:
        if ex.response is not None and 'already exists' in ex.response.content.decode('utf-8'):
            pass
    
    expected_vols = {}
    for scale in range(1+max_scale):
        if scale == 0:
            scaled_vol = random_segmentation
        else:
            scaled_vol = downsample(scaled_vol, 2, 'labels-numba')
        post_labelmap_voxels(dvid_address, repo_uuid, input_segmentation_name, (0,0,0), scaled_vol, scale=scale)
        expected_vols[scale] = scaled_vol
    
    template_dir = tempfile.mkdtemp(suffix="labelmapcopy-template")
 
    config_text = textwrap.dedent(f"""\
        workflow-name: labelmapcopy
        cluster-type: {CLUSTER_TYPE}
         
        input:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {input_segmentation_name}
            supervoxels: true
           
          geometry:
            message-block-shape: [512,64,64]
            available-scales: [0,1,2]
 
        output:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {output_segmentation_name}
            supervoxels: true
            disable-indexing: true
            create-if-necessary: true
    """)
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)
 
    return template_dir, config, expected_vols, dvid_address, repo_uuid, output_segmentation_name


def test_labelmapcopy(setup_dvid_segmentation_input, disable_auto_retry):
    template_dir, _config, expected_vols, dvid_address, repo_uuid, output_segmentation_name = setup_dvid_segmentation_input
    
    _execution_dir, workflow = launch_flow(template_dir, 1)
    final_config = workflow.config

    output_box_xyz = np.array( final_config['output']['geometry']['bounding-box'] )
    output_box_zyx = output_box_xyz[:,::-1]
    
    max_scale = final_config['labelmapcopy']['max-scale']
    for scale in range(1+max_scale):
        scaled_box = output_box_zyx // (2**scale)
        output_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, scaled_box, scale=scale)
        assert (output_vol == expected_vols[scale]).all(), \
            f"Written vol does not match expected for scale {scale}"


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")
    
    #from neuclease import configure_default_logging
    #configure_default_logging()
    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_labelmapcopy']
    pytest.main(args)
