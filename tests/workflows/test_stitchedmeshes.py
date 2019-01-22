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

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchflow import launch_flow

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
    input_segmentation_name = 'segmentation-stitchedmesh-input'

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

    object_boxes = []
    for label, corner in zip([100,200,300], [(256,256,256), (512,512,512), (1024,1024,1024)]):
        box = place_test_object(label, corner)
        object_boxes.append( box )
     
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
          bodies: [100,200,250,300] # 250 doesn't exist -- error should be noted but shouldn't kill the job
          concurrent-bodies: 2
          scale: 1
          block-halo: 1
          stitch: True
          smoothing-iterations: 3
          decimation-fraction: 0.01
          format: obj
          output-directory: meshes
          skip-existing: false
          error-mode: warn
    """)
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)
 
    return template_dir, config, dvid_address, repo_uuid, object_boxes


def test_stitchedmeshes(setup_dvid_segmentation_input):
    template_dir, _config, _dvid_address, _repo_uuid, object_boxes = setup_dvid_segmentation_input
    
    execution_dir, _workflow = launch_flow(template_dir, 1)
    #final_config = workflow.config

    assert os.path.exists(f"{execution_dir}/meshes/100.obj")
    assert os.path.exists(f"{execution_dir}/meshes/200.obj")
    assert os.path.exists(f"{execution_dir}/meshes/300.obj")
    
    # Make sure the mesh vertices appeared in the right place.
    # (If they weren't rescaled, this won't work.)
    mesh100 = Mesh.from_file(f"{execution_dir}/meshes/100.obj")
    assert (mesh100.vertices_zyx[:] >= object_boxes[0][0]).all()
    assert (mesh100.vertices_zyx[:] <= object_boxes[0][1]).all()
    
    mesh200 = Mesh.from_file(f"{execution_dir}/meshes/200.obj")
    assert (mesh200.vertices_zyx[:] >= object_boxes[1][0]).all()
    assert (mesh200.vertices_zyx[:] <= object_boxes[1][1]).all()
    
    mesh300 = Mesh.from_file(f"{execution_dir}/meshes/300.obj")
    assert (mesh300.vertices_zyx[:] >= object_boxes[2][0]).all()
    assert (mesh300.vertices_zyx[:] <= object_boxes[2][1]).all()
    
    # Here's where our test meshes ended up:
    #print(f"{execution_dir}/meshes/100.obj")
    #print(f"{execution_dir}/meshes/200.obj")
    #print(f"{execution_dir}/meshes/300.obj")
    #print(f'{execution_dir}/mesh-stats.csv')
    
    df = pd.read_csv(f'{execution_dir}/mesh-stats.csv')
    assert len(df) == 4
    df.set_index('body', inplace=True)

    assert df.loc[100, 'result'] == 'success'
    assert df.loc[200, 'result'] == 'success'
    assert df.loc[250, 'result'] == 'error'  # intentional error
    assert df.loc[300, 'result'] == 'success'

if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")

    import flyemflows
    os.chdir(os.path.dirname(flyemflows.__file__))    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_stitchedmeshes'])
