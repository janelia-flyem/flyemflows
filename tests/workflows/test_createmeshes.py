import os
import tempfile
import textwrap
from io import StringIO

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from neuclease.dvid import create_labelmap_instance, post_labelmap_voxels

from vol2mesh import Mesh

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchflow import launch_flow
from neuclease.util.box import box_to_slicing

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

def create_test_segmentation():
    if os.environ.get('TRAVIS', '') == 'true':
        # On Travis-CI, store this test data in a place that gets cached.
        d = '/home/travis/miniconda/test-data'
    else:
        d = '/tmp'

    vol_path = f'{d}/test-createmeshes-segmentation.npy'
    boxes_path = f'{d}/test-createmeshes-boxes.npy'
    if os.path.exists(vol_path):
        return np.load(vol_path), np.load(boxes_path)

    test_volume = np.zeros((256, 256, 256), np.uint64)

    def place_test_object(label, corner):
        corner = np.array(corner)
        object_vol = create_test_object().astype(np.uint64)
        object_vol *= label
        object_box = np.array([corner, corner + object_vol.shape])
        
        testvol_view = test_volume[box_to_slicing(*object_box)]
        testvol_view[:] = np.where(object_vol, object_vol, testvol_view)
        return object_box

    # Place four text objects
    object_boxes = []
    labels = [100,200,300]
    corners = [(10,10,10), (10, 60, 10), (10, 110, 10)]
    for label, corner in zip(labels, corners):
        box = place_test_object(label, corner)
        object_boxes.append( box )

    np.save(vol_path, test_volume) # Cache for next pytest run
    np.save(boxes_path, np.array(object_boxes))
    return test_volume, object_boxes


@pytest.fixture(scope='module')
def setup_segmentation_input(setup_dvid_repo):
    dvid_address, repo_uuid = setup_dvid_repo
    input_segmentation_name = 'segmentation-createmeshes-input'
    test_volume, object_boxes = create_test_segmentation()
 
    create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name, max_scale=3)
    post_labelmap_voxels(dvid_address, repo_uuid, input_segmentation_name, (0,0,0), test_volume, downres=True, noindexing=False)
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
            supervoxels: true
           
          geometry:
            block-width: 64
            available-scales: [0,1,2,3]
 
        output:
          directory: meshes

        createmeshes:
          subset-supervoxels: []
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
    print(execution_dir)
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
