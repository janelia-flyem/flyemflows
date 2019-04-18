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

from vol2mesh import Mesh
from confiddler import dump_config

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchflow import launch_flow

yaml = YAML()
yaml.default_flow_style = False

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


@pytest.fixture(scope='module')
def setup_segmentation_input(setup_dvid_repo):
    dvid_address, repo_uuid = setup_dvid_repo
    input_segmentation_name = 'segmentation-createmeshes-input'
 
    create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name, max_scale=3)

    def place_test_object(label, corner):
        corner = np.array(corner)
        label_vol = create_test_object().astype(np.uint64)
        label_vol *= label
        
        corners = np.array(list(ndrange((0,0,0), label_vol.shape, (64,64,64))))
        
        # Drop it away from (0,0,0) to make sure the workflow handles arbitrary locations
        corners += corner
        
        blockwise_vol = view_as_blocks(label_vol, (64,64,64))
        blocks = blockwise_vol.reshape(-1,64,64,64)
        
        # post to dvid
        post_labelarray_blocks(dvid_address, repo_uuid, input_segmentation_name, corners, blocks, downres=True, noindexing=False)
        return np.array([corner, corner + label_vol.shape])

    # Place four text objects
    object_boxes = []
    labels = [100,200,300]
    corners = [(256,256,256), (512,512,512), (1024,1024,1024)]
    for label, corner in zip(labels, corners):
        box = place_test_object(label, corner)
        object_boxes.append( box )

    return dvid_address, repo_uuid, input_segmentation_name, object_boxes


@pytest.fixture
def setup_createmeshes_config(setup_segmentation_input, disable_auto_retry):
    dvid_address, repo_uuid, input_segmentation_name, object_boxes = setup_segmentation_input
    template_dir = tempfile.mkdtemp(suffix="createmeshes-template")

    config_text = textwrap.dedent(f"""\
        workflow-name: createmeshes
        cluster-type: {CLUSTER_TYPE}

        input:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {input_segmentation_name}
           
          geometry:
            block-width: 64
            available-scales: [0,1,2,3]
 
        output:
          directory: meshes

        createmeshes:
          subset-labels: []
          halo: 0
          
          pre-stitch-parameters:
            smoothing: 0
            decimation: 1.0
            compute-normals: false

          post-stitch-parameters:
            smoothing: 0
            decimation: 1.0
            compute-normals: false
        
          size-filters:
            minimum-supervoxel-size: 0
            maximum-supervoxel-size: 1e9
            minimum-body-size: 0
            maximum-body-size: 1e9

          max-body-vertices: null
          rescale-before-write: 1.0
          format: obj
          include-empty: false
          skip-existing: false
    """)
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)
 
    return template_dir, config, dvid_address, repo_uuid, object_boxes


def test_createmeshes(setup_createmeshes_config, disable_auto_retry):
    template_dir, _config, _dvid_address, _repo_uuid, object_boxes = setup_createmeshes_config
    
    execution_dir, _workflow = launch_flow(template_dir, 1)
    #final_config = workflow.config

    assert os.path.exists(f"{execution_dir}/meshes/100.obj")
    assert os.path.exists(f"{execution_dir}/meshes/200.obj")
    assert os.path.exists(f"{execution_dir}/meshes/300.obj")
    

if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")

    import flyemflows
    os.chdir(os.path.dirname(flyemflows.__file__))    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_createmeshes'])
