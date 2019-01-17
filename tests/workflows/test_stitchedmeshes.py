import os
import tempfile
import textwrap
from io import StringIO

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.util import view_as_blocks 

from neuclease.util import ndrange
from neuclease.dvid import create_labelmap_instance, post_labelarray_blocks

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchworkflow import launch_workflow

TESTVOL_SHAPE = (256,256,256)

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')

def create_test_object():
    # Create a test object (shaped like an 'X')
    center_line_img = np.zeros((128,128,128), dtype=np.uint32)
    for i in range(128):
        center_line_img[i, i, i] = 1
        center_line_img[127-i, i, i] = 1
    
    # Scipy distance_transform_edt conventions are opposite of vigra:
    # it calculates distances of non-zero pixels to the zero pixels.
    center_line_img = 1 - center_line_img
    distance_to_line = distance_transform_edt(center_line_img)
    binary_vol = (distance_to_line <= 10).astype(np.uint8)
    return binary_vol


@pytest.fixture
def setup_dvid_segmentation_input(setup_dvid_repo):
    dvid_address, repo_uuid = setup_dvid_repo
    input_segmentation_name = 'segmentation-input'

    create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name, max_scale=3)
     
    label_vol = create_test_object().astype(np.uint64)
    label_vol *= 100
    
    corners = np.array(list(ndrange((0,0,0), label_vol.shape, (64,64,64))))
    
    # Drop it away from (0,0,0) to make sure the workflow handles arbitrary locations
    corners += 256
    
    blockwise_vol = view_as_blocks(label_vol, (64,64,64))
    blocks = blockwise_vol.reshape(-1,64,64,64)
    
    # post to dvid
    post_labelarray_blocks(dvid_address, repo_uuid, input_segmentation_name, corners, blocks, downres=True, noindexing=False)
     
    template_dir = tempfile.mkdtemp(suffix="stitchedmeshes-template")
 
    config_text = textwrap.dedent(f"""\
        workflow-name: stitchedmeshes
        cluster-type: {CLUSTER_TYPE}

        input:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {input_segmentation_name}
           
          geometry:
            bounding-box: [[0,0,0], [128,128,128]]
            block-width: 64
            available-scales: [0,1,2,3]
 
        stitchedmeshes:
          bodies: [100]
          #bodies: [100, 200] # 200 doesn't exist, will fail (see below)
          scale: 0
          block-halo: 1
          stitch: True
          smoothing-iterations: 3
          decimation-fraction: 0.01
          format: obj
          output-directory: meshes
          skip-existing: false
    """)
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)
 
    return template_dir, config, dvid_address, repo_uuid


def test_stitchedmeshes(setup_dvid_segmentation_input):
    template_dir, _config, _dvid_address, _repo_uuid = setup_dvid_segmentation_input
    
    execution_dir, _workflow = launch_workflow(template_dir, 1)
    #final_config = workflow.config

    assert os.path.exists(f"{execution_dir}/meshes/100.obj")
    
    # Here's where our test mesh ended up:
    print(f"{execution_dir}/meshes/100.obj")
    
#     df = pd.read_csv(f'{execution_dir}/mesh-stats.csv')
#     assert len(df) == 2
#     assert df.loc[0, 'body'] == 100
#     assert df.loc[0, 'scale'] == 1
#     assert df.loc[0, 'result'] == 'success'
#     
    # The second body didn't exist, so it fails (but doesn't kill the workflow)
#     assert df.loc[1, 'body'] == 200
#     assert df.loc[1, 'scale'] == 0
#     assert df.loc[1, 'result'] == 'error-sparsevol-coarse'


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")

    import flyemflows
    os.chdir(os.path.dirname(flyemflows.__file__))    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_stitchedmeshes'])
