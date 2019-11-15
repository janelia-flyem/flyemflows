import os
import pickle
import tempfile
import textwrap
from io import StringIO

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from neuclease.dvid import create_labelmap_instance, post_labelmap_voxels, fetch_supervoxel, fetch_key

from vol2mesh import Mesh

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchflow import launch_flow
from neuclease.util.box import box_to_slicing

yaml = YAML()
yaml.default_flow_style = False

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')


def create_test_object(height=128, radius=10):
    """
    Create a test object (shaped like an 'X')
    with overall height (and width and depth) determined by 'height',
    and whose "arm" thickness is determined by the given radius.
    """
    center_line_img = np.zeros((height,height,height), dtype=np.uint32)
    for i in range(height):
        center_line_img[i, i, i] = 1
        center_line_img[height-1-i, i, i] = 1
    
    # Scipy distance_transform_edt conventions are opposite of vigra:
    # it calculates distances of non-zero pixels to the zero pixels.
    center_line_img = 1 - center_line_img
    distance_to_line = distance_transform_edt(center_line_img)
    binary_vol = (distance_to_line <= radius).astype(np.uint8)
    return binary_vol


def create_test_segmentation():
    if os.environ.get('TRAVIS', '') == 'true':
        # On Travis-CI, store this test data in a place that gets cached.
        d = '/home/travis/miniconda/test-data'
    else:
        d = '/tmp'

    vol_path = f'{d}/test-createmeshes-segmentation.npy'
    boxes_path = f'{d}/test-createmeshes-boxes.pkl'
    sizes_path = f'{d}/test-createmeshes-sizes.pkl'
    if os.path.exists(vol_path):
        test_volume = np.load(vol_path)
        object_boxes = pickle.load(open(boxes_path, 'rb'))
        object_sizes = pickle.load(open(sizes_path, 'rb'))
        return test_volume, object_boxes, object_sizes

    test_volume = np.zeros((256, 256, 256), np.uint64)

    def place_test_object(label, corner, height):
        corner = np.array(corner)
        object_vol = create_test_object(height).astype(np.uint64)
        object_vol *= label
        object_box = np.array([corner, corner + object_vol.shape])
        
        testvol_view = test_volume[box_to_slicing(*object_box)]
        testvol_view[:] = np.where(object_vol, object_vol, testvol_view)
        return object_box, (object_vol != 0).sum()

    # Place four text objects
    object_boxes = {}
    object_sizes = {}
    labels = [100,200,300]
    corners = [(10,10,10), (10, 60, 10), (10, 110, 10)]
    heights = (200, 150, 50)
    for label, corner, height in zip(labels, corners, heights):
        box, num_voxels = place_test_object(label, corner, height)
        object_boxes[label] = box
        object_sizes[label] = int(num_voxels)

    np.save(vol_path, test_volume) # Cache for next pytest run
    pickle.dump(object_boxes, open(boxes_path, 'wb'))
    pickle.dump(object_sizes, open(sizes_path, 'wb'))
    return test_volume, object_boxes, object_sizes


@pytest.fixture(scope='module')
def setup_segmentation_input(setup_dvid_repo):
    dvid_address, repo_uuid = setup_dvid_repo
    input_segmentation_name = 'segmentation-createmeshes-input'
    test_volume, object_boxes, object_sizes = create_test_segmentation()
 
    create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name, max_scale=3)
    post_labelmap_voxels(dvid_address, repo_uuid, input_segmentation_name, (0,0,0), test_volume, downres=True, noindexing=False)
    return dvid_address, repo_uuid, input_segmentation_name, object_boxes, object_sizes


@pytest.fixture
def setup_createmeshes_config(setup_segmentation_input, disable_auto_retry):
    dvid_address, repo_uuid, input_segmentation_name, object_boxes, object_sizes = setup_segmentation_input
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
 
    return template_dir, config, dvid_address, repo_uuid, object_boxes, object_sizes


def check_outputs(execution_dir, object_boxes, subset_labels=None, stats_dir=None):
    """
    Basic checks to make sure the meshes for the given labels were
    generated and not terribly wrong.
    """
    stats_dir = stats_dir or execution_dir
    
    # Check all test objects by default.
    if subset_labels is None:
        subset_labels = sorted(object_boxes.keys())

    df = pd.DataFrame( np.load(f'{stats_dir}/final-mesh-stats.npy') )
    assert len(df) == len(subset_labels)
    df.set_index('sv', inplace=True)

    for label in subset_labels:
        assert df.loc[label, 'file_size'] > 0

        # Here's where our test meshes ended up:
        #print(f"{execution_dir}/meshes/{label}.obj")
        assert os.path.exists(f"{execution_dir}/meshes/{label}.obj")
    
        # Make sure the mesh vertices appeared in the right place.
        # (If they weren't rescaled, this won't work.)
        mesh = Mesh.from_file(f"{execution_dir}/meshes/{label}.obj")
        assert np.allclose(mesh.vertices_zyx.min(axis=0), object_boxes[label][0], 1)
        assert np.allclose(mesh.vertices_zyx.max(axis=0), object_boxes[label][1], 1)


def test_createmeshes_basic(setup_createmeshes_config, disable_auto_retry):
    template_dir, _config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
     
    execution_dir, _workflow = launch_flow(template_dir, 1)
    #print(execution_dir)
    check_outputs(execution_dir, object_boxes)

def test_createmeshes_to_tarsupervoxels(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
     
    tsv_instance = 'test_createmeshes_to_tarsupervoxels'
    config['output'] = {'tarsupervoxels':
                            {'instance': tsv_instance,
                             'create-if-necessary': True}}
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    _execution_dir, _workflow = launch_flow(template_dir, 1)
    
    server = config['input']['dvid']['server']
    uuid = config['input']['dvid']['uuid']
    for sv in [100,200,300]:
        mesh_bytes = fetch_supervoxel(server, uuid, tsv_instance, sv)
        mesh = Mesh.from_buffer(mesh_bytes, fmt='obj')
        assert np.allclose(mesh.vertices_zyx.min(axis=0), object_boxes[sv][0], 1)
        assert np.allclose(mesh.vertices_zyx.max(axis=0), object_boxes[sv][1], 1)


def test_createmeshes_to_keyvalue(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
     
    kv_instance = 'test_createmeshes_to_keyvalue'
    config['output'] = {'keyvalue':
                            {'instance': kv_instance,
                             'create-if-necessary': True}}
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    _execution_dir, _workflow = launch_flow(template_dir, 1)
    
    server = config['input']['dvid']['server']
    uuid = config['input']['dvid']['uuid']
    for sv in [100,200,300]:
        mesh_bytes = fetch_key(server, uuid, kv_instance, f'{sv}.obj')
        mesh = Mesh.from_buffer(mesh_bytes, fmt='obj')
        assert np.allclose(mesh.vertices_zyx.min(axis=0), object_boxes[sv][0], 1)
        assert np.allclose(mesh.vertices_zyx.max(axis=0), object_boxes[sv][1], 1)


def test_createmeshes_subset_svs(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
    config['createmeshes']['subset-supervoxels'] = [100,300]
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))
     
    execution_dir, _workflow = launch_flow(template_dir, 1)
 
    df = pd.DataFrame( np.load(f'{execution_dir}/final-mesh-stats.npy') )
    assert 200 not in df['sv'].values
 
    check_outputs(execution_dir, object_boxes, subset_labels=[100,300])


def test_createmeshes_bad_subset_svs(setup_createmeshes_config, disable_auto_retry):
    """
    If one (or more) of the listed supervoxels no longer exists,
    DVID will return a 0 for that supervoxel's mapping,
    and its sparsevol cannot be determined.
    The job should complete anyway, for the other supervoxels.
    """
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
    config['createmeshes']['subset-supervoxels'] = [100,999]
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))
    
    execution_dir, _workflow = launch_flow(template_dir, 1)

    # List of bad svs is reported in this CSV file.
    # See DvidVolumeService.sparse_brick_coords_for_labels()
    assert pd.read_csv(f'{execution_dir}/labels-without-sparsevols.csv')['sv'].tolist() == [999]

    df = pd.DataFrame( np.load(f'{execution_dir}/final-mesh-stats.npy') )
    assert 100 in df['sv'].values
    assert 200 not in df['sv'].values
    assert 300 not in df['sv'].values
    assert 999 not in df['sv'].values

    check_outputs(execution_dir, object_boxes, subset_labels=[100])


def test_createmeshes_subset_bodies(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
    config['createmeshes']['subset-bodies'] = [100,300]
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))
    
    execution_dir, _workflow = launch_flow(template_dir, 1)

    df = pd.DataFrame( np.load(f'{execution_dir}/final-mesh-stats.npy') )
    assert 200 not in df['sv'].values

    check_outputs(execution_dir, object_boxes, subset_labels=[100,300])


def test_createmeshes_bad_subset_bodies(setup_createmeshes_config, disable_auto_retry):
    """
    If one (or more) of the listed bodies no longer exists,
    DVID returns a 404 when we try to fetch its supervoxels.
    The job should complete anyway, for the other bodies.
    """
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
    config['createmeshes']['subset-bodies'] = [100,999] # 999 doesn't exist
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))
    
    execution_dir, _workflow = launch_flow(template_dir, 1)

    assert pd.read_csv(f'{execution_dir}/missing-bodies.csv')['body'].tolist() == [999]

    df = pd.DataFrame( np.load(f'{execution_dir}/final-mesh-stats.npy') )
    assert 100 in df['sv'].values
    assert 200 not in df['sv'].values
    assert 300 not in df['sv'].values
    assert 999 not in df['sv'].values

    check_outputs(execution_dir, object_boxes, subset_labels=[100])


def test_createmeshes_subset_bodies_in_batches(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
    config['createmeshes']['subset-bodies'] = [100,200,300]
    config['createmeshes']['subset-batch-size'] = 2
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))
    
    execution_dir, _workflow = launch_flow(template_dir, 1)

    check_outputs(execution_dir, object_boxes, subset_labels=[100,200], stats_dir=f'{execution_dir}/batch-00')
    check_outputs(execution_dir, object_boxes, subset_labels=[300], stats_dir=f'{execution_dir}/batch-01')


def test_createmeshes_filter_supervoxels(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, object_sizes = setup_createmeshes_config

    # Set size filter to exclude the largest SV (100) and smallest SV (300),
    # leaving only the middle object (SV 200).
    assert object_sizes[300] < object_sizes[200] < object_sizes[100]
    config['createmeshes']['size-filters']['minimum-supervoxel-size'] = object_sizes[300]+1
    config['createmeshes']['size-filters']['maximum-supervoxel-size'] = object_sizes[100]-1
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))
    
    execution_dir, _workflow = launch_flow(template_dir, 1)

    df = pd.DataFrame( np.load(f'{execution_dir}/final-mesh-stats.npy') )
    assert 100 not in df['sv'].values
    assert 300 not in df['sv'].values

    check_outputs(execution_dir, object_boxes, subset_labels=[200])


def test_createmeshes_rescale_isotropic(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
    config['createmeshes']['rescale-before-write'] = 2
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))
     
    execution_dir, _workflow = launch_flow(template_dir, 1)

    scaled_boxes = { label: box * 2 for label, box in object_boxes.items() }
    check_outputs(execution_dir, scaled_boxes)


def test_createmeshes_rescale_anisotropic(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
    config['createmeshes']['rescale-before-write'] = [2,3,4]
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))
     
    execution_dir, _workflow = launch_flow(template_dir, 1)

    # Note xyz vs zyx
    scaled_boxes = { label: box * [4,3,2] for label, box in object_boxes.items() }
    check_outputs(execution_dir, scaled_boxes)


def test_createmeshes_skip_existing(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
    config['createmeshes']['skip-existing'] = True
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    # Create an empty file for mesh 200
    os.makedirs(f"{template_dir}/meshes")
    open(f"{template_dir}/meshes/200.obj", 'wb').close()
    execution_dir, _workflow = launch_flow(template_dir, 1)
 
    # The file should have been left alone (still empty).
    assert open(f"{execution_dir}/meshes/200.obj", 'rb').read() == b''

    # But other meshes were generated.
    check_outputs(execution_dir, object_boxes, subset_labels=[100,300])


def test_createmeshes_from_body_source(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
    config["input"]["dvid"]["supervoxels"] = False
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    execution_dir, _workflow = launch_flow(template_dir, 1)
    #print(execution_dir)

    # In this test, each supervoxel is its own body anyway.
    check_outputs(execution_dir, object_boxes)

def test_createmeshes_from_body_source_subset_bodies(setup_createmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_createmeshes_config
    
    config["input"]["dvid"].update({
        "supervoxels": False
    })

    config["input"]["geometry"].update({
        "message-block-shape": [128,128,128]
    })

    config["input"]["adapters"] = {
        "rescale-level": 1
    }

    config["createmeshes"].update({
        "subset-bodies": [100,300],
        "rescale-before-write": 2.0
    })

    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    execution_dir, _workflow = launch_flow(template_dir, 1)
    #print(execution_dir)

    # In this test, each supervoxel is its own body anyway.
    check_outputs(execution_dir, object_boxes, [100,300])


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")

    import flyemflows
    os.chdir(os.path.dirname(flyemflows.__file__))
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_createmeshes']
    #args += ['-x']
    #args += ['-Werror']
    #args += ['-k', 'createmeshes_subset_bodies_in_batches']
    pytest.main(args)
