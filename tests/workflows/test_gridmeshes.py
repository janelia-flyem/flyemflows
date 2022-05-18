import os
import copy
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
from neuclease.util import box_to_slicing, extract_subvol, overwrite_subvol, boxes_from_grid, place_sphere, region_features

yaml = YAML()
yaml.default_flow_style = False

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')


def create_test_segmentation():
    """
    Create a 'gridded' test segmentation.
    The final volume will have shape (256,256,256),
    with a grid of (128,128,128).

    The output will be consecutively numbered,
    with labels 1..17
    """
    vol = np.zeros((256, 256, 256), np.uint64)
    place_sphere(vol, (128,128,128), 50, 1)
    place_sphere(vol, (200, 128, 128), 50, 2)
    place_sphere(vol, (0,0,0), 60, 3)
    place_sphere(vol, (0,100,0), 75, 4)
    place_sphere(vol, (178, 216, 47), 75, 5)

    # Create a 'gridded' segmentation
    N = vol.max()
    for i, box in enumerate(boxes_from_grid([(0,0,0), vol.shape], (128, 128, 128))):
        subvol = extract_subvol(vol, box)
        subvol = np.where(subvol, subvol + i*N, 0)
        overwrite_subvol(vol, box, subvol)

    # Renumber it consecutively just to make it easier to remember
    # which labels are present in this test volume.
    labels = pd.unique(vol.ravel())
    labels.sort()
    remap = np.zeros(int(labels.max()+1), dtype=np.uint64)
    remap[labels] = np.arange(len(labels), dtype=np.uint64)
    vol = remap[vol]

    feat = region_features(vol)
    return vol, feat['Box'].to_dict(), feat['Count'].to_dict()

@pytest.fixture(scope='module')
def setup_segmentation_input(setup_dvid_repo):
    dvid_address, repo_uuid = setup_dvid_repo
    input_segmentation_name = 'segmentation-gridmeshes-input'
    test_volume, object_boxes, object_sizes = create_test_segmentation()

    create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name, max_scale=3)
    post_labelmap_voxels(dvid_address, repo_uuid, input_segmentation_name, (0,0,0), test_volume, downres=True, noindexing=False)
    return dvid_address, repo_uuid, input_segmentation_name, object_boxes, object_sizes


@pytest.fixture
def setup_gridmeshes_config(setup_segmentation_input, disable_auto_retry):
    dvid_address, repo_uuid, input_segmentation_name, object_boxes, object_sizes = setup_segmentation_input
    template_dir = tempfile.mkdtemp(suffix="gridmeshes-template")

    config_text = textwrap.dedent(f"""\
        workflow-name: gridmeshes
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
            message-block-shape: [128,128,128]

          adapters:
            rescale-level: 0

        output:
          directory: meshes

        gridmeshes:
          slab-shape: [128, 256, 256]
          subset-supervoxels: []

          mesh-parameters:
            smoothing: 0
            decimation: 1.0
            compute-normals: false

          minimum-supervoxel-size: 0
          maximum-supervoxel-size: 1e9

          rescale-before-write: 1.0
          format: obj
          skip-existing: false
    """)

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)

    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)

    return template_dir, config, dvid_address, repo_uuid, object_boxes, object_sizes


def check_outputs(execution_dir, object_boxes, subset_labels=None, stats_dir=None, max_vertices=None, atol=[2,2,2]):
    """
    Basic checks to make sure the meshes for the given labels were
    generated and not terribly wrong.
    """
    stats_dir = stats_dir or execution_dir

    # Check all test objects by default.
    if subset_labels is None:
        subset_labels = sorted(object_boxes.keys())

    # df = pd.DataFrame( np.load(f'{stats_dir}/final-mesh-stats.npy') )
    # assert len(df) == len(subset_labels)
    # df.set_index('sv', inplace=True)

    for label in subset_labels:
        # assert df.loc[label, 'file_size'] > 0

        # Here's where our test meshes ended up:
        #print(f"{execution_dir}/meshes/{label}.obj")
        assert os.path.exists(f"{execution_dir}/meshes/{label}.obj")

        # Make sure the mesh vertices appeared in the right place.
        # (If they weren't rescaled, this won't work.)
        # We use a very loose tolerance to allow for 'cuffs'.
        mesh = Mesh.from_file(f"{execution_dir}/meshes/{label}.obj")
        assert np.allclose(mesh.vertices_zyx.min(axis=0), object_boxes[label][0], atol=atol)
        assert np.allclose(mesh.vertices_zyx.max(axis=0), object_boxes[label][1], atol=atol)

        if max_vertices is not None:
            assert len(mesh.vertices_zyx) <= 1.1*max_vertices


def test_gridmeshes_basic(setup_gridmeshes_config, disable_auto_retry):
    template_dir, _config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config

    execution_dir, _workflow = launch_flow(template_dir, 1)
    #print(execution_dir)
    check_outputs(execution_dir, object_boxes)


# def test_gridmeshes_max_vertices(setup_gridmeshes_config, disable_auto_retry):
#     template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
#     config = copy.deepcopy(config)
#     max_vertices = 1000
#     config['gridmeshes']['post-stitch-parameters']['max-vertices'] = max_vertices
#     YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

#     execution_dir, _workflow = launch_flow(template_dir, 1)
#     #print(execution_dir)
#     check_outputs(execution_dir, object_boxes, max_vertices=max_vertices)


# def test_gridmeshes_max_svs_per_brick(setup_gridmeshes_config, disable_auto_retry):
#     template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
#     config = copy.deepcopy(config)
#     config['gridmeshes']['max-svs-per-brick'] = 1
#     # Use just one brick, but it should end up getting split for each object.
#     config["input"]["geometry"].update({
#         "message-block-shape": [128,128,128]
#     })
#     YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

#     execution_dir, _workflow = launch_flow(template_dir, 1)
#     #print(execution_dir)
#     check_outputs(execution_dir, object_boxes)


def test_gridmeshes_to_tarsupervoxels(setup_gridmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
    config = copy.deepcopy(config)

    tsv_instance = 'test_gridmeshes_to_tarsupervoxels'
    config['output'] = {'tarsupervoxels':
                            {'instance': tsv_instance,
                             'create-if-necessary': True}}
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    _execution_dir, _workflow = launch_flow(template_dir, 1)

    server = config['input']['dvid']['server']
    uuid = config['input']['dvid']['uuid']
    for sv in object_boxes.keys():
        mesh_bytes = fetch_supervoxel(server, uuid, tsv_instance, sv)
        mesh = Mesh.from_buffer(mesh_bytes, fmt='obj')

        # We use a very loose tolerance to allow for 'cuffs'.
        assert np.allclose(mesh.vertices_zyx.min(axis=0), object_boxes[sv][0], atol=2)
        assert np.allclose(mesh.vertices_zyx.max(axis=0), object_boxes[sv][1], atol=2)


def test_gridmeshes_to_keyvalue(setup_gridmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
    config = copy.deepcopy(config)

    kv_instance = 'test_gridmeshes_to_keyvalue'
    config['output'] = {'keyvalue':
                            {'instance': kv_instance,
                             'create-if-necessary': True}}
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    _execution_dir, _workflow = launch_flow(template_dir, 1)

    server = config['input']['dvid']['server']
    uuid = config['input']['dvid']['uuid']
    for sv in object_boxes.keys():
        mesh_bytes = fetch_key(server, uuid, kv_instance, f'{sv}.obj')
        mesh = Mesh.from_buffer(mesh_bytes, fmt='obj')

        # We use a very loose tolerance to allow for 'cuffs'.
        assert np.allclose(mesh.vertices_zyx.min(axis=0), object_boxes[sv][0], atol=2)
        assert np.allclose(mesh.vertices_zyx.max(axis=0), object_boxes[sv][1], atol=2)


def test_gridmeshes_subset_svs(setup_gridmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
    config = copy.deepcopy(config)

    config['gridmeshes']['subset-supervoxels'] = [5, 10, 15]
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    execution_dir, _workflow = launch_flow(template_dir, 1)

    assert not os.path.exists(f"{execution_dir}/meshes/1.obj")
    assert not os.path.exists(f"{execution_dir}/meshes/2.obj")
    assert not os.path.exists(f"{execution_dir}/meshes/3.obj")

    check_outputs(execution_dir, object_boxes, subset_labels=[5, 10, 15])


# def test_gridmeshes_bad_subset_svs(setup_gridmeshes_config, disable_auto_retry):
#     """
#     If one (or more) of the listed supervoxels no longer exists,
#     DVID will return a 0 for that supervoxel's mapping,
#     and its sparsevol cannot be determined.
#     The job should complete anyway, for the other supervoxels.
#     """
#     template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
#     config = copy.deepcopy(config)
#
#     config['gridmeshes']['subset-supervoxels'] = [1,999]
#     YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

#     execution_dir, _workflow = launch_flow(template_dir, 1)

#     # List of bad svs is reported in this CSV file.
#     # See DvidVolumeService.sparse_brick_coords_for_labels()
#     assert pd.read_csv(f'{execution_dir}/labels-without-sparsevols.csv')['sv'].tolist() == [999]

#     df = pd.DataFrame( np.load(f'{execution_dir}/final-mesh-stats.npy') )
#     assert 100 in df['sv'].values
#     assert 200 not in df['sv'].values
#     assert 300 not in df['sv'].values
#     assert 999 not in df['sv'].values

#     check_outputs(execution_dir, object_boxes, subset_labels=[1])


def test_gridmeshes_subset_bodies(setup_gridmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
    config = copy.deepcopy(config)

    config['gridmeshes']['subset-bodies'] = [5, 10, 15]
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    execution_dir, _workflow = launch_flow(template_dir, 1)

    assert not os.path.exists(f"{execution_dir}/meshes/1.obj")
    assert not os.path.exists(f"{execution_dir}/meshes/2.obj")
    assert not os.path.exists(f"{execution_dir}/meshes/3.obj")

    check_outputs(execution_dir, object_boxes, subset_labels=[5, 10, 15])


# def test_gridmeshes_bad_subset_bodies(setup_gridmeshes_config, disable_auto_retry):
#     """
#     If one (or more) of the listed bodies no longer exists,
#     DVID returns a 404 when we try to fetch its supervoxels.
#     The job should complete anyway, for the other bodies.
#     """
#     template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
#     config = copy.deepcopy(config)
#
#     config['gridmeshes']['subset-bodies'] = [1,999] # 999 doesn't exist
#     YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

#     execution_dir, _workflow = launch_flow(template_dir, 1)

#     assert pd.read_csv(f'{execution_dir}/missing-bodies.csv')['body'].tolist() == [999]

#     df = pd.DataFrame( np.load(f'{execution_dir}/final-mesh-stats.npy') )
#     assert 100 in df['sv'].values
#     assert 200 not in df['sv'].values
#     assert 300 not in df['sv'].values
#     assert 999 not in df['sv'].values

#     check_outputs(execution_dir, object_boxes, subset_labels=[100])


# def test_gridmeshes_subset_bodies_in_batches(setup_gridmeshes_config, disable_auto_retry):
#     template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
#     config = copy.deepcopy(config)
#
#     config['gridmeshes']['subset-bodies'] = [100,200,300]
#     config['gridmeshes']['subset-batch-size'] = 2
#     YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

#     execution_dir, _workflow = launch_flow(template_dir, 1)

#     #print(execution_dir)
#     check_outputs(execution_dir, object_boxes, subset_labels=[100,200], stats_dir=f'{execution_dir}/batch-00')
#     check_outputs(execution_dir, object_boxes, subset_labels=[300], stats_dir=f'{execution_dir}/batch-01')


# def test_gridmeshes_filter_supervoxels(setup_gridmeshes_config, disable_auto_retry):
#     template_dir, config, _dvid_address, _repo_uuid, object_boxes, object_sizes = setup_gridmeshes_config
#     config = copy.deepcopy(config)
#
#     # Set size filter to exclude the largest SV (100) and smallest SV (300),
#     # leaving only the middle object (SV 200).
#     assert object_sizes[300] < object_sizes[200] < object_sizes[100]
#     config['gridmeshes']['size-filters']['minimum-supervoxel-size'] = object_sizes[300]+1
#     config['gridmeshes']['size-filters']['maximum-supervoxel-size'] = object_sizes[100]-1
#     YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

#     execution_dir, _workflow = launch_flow(template_dir, 1)

#     df = pd.DataFrame( np.load(f'{execution_dir}/final-mesh-stats.npy') )
#     assert 100 not in df['sv'].values
#     assert 300 not in df['sv'].values

#     check_outputs(execution_dir, object_boxes, subset_labels=[200])


def test_gridmeshes_rescale(setup_gridmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
    config = copy.deepcopy(config)

    config['input']['adapters']['rescale-level'] = 1
    config['gridmeshes']['rescale-before-write'] = 2
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    execution_dir, _workflow = launch_flow(template_dir, 1)

    scaled_boxes = { label: box for label, box in object_boxes.items() }
    check_outputs(execution_dir, scaled_boxes)


def test_gridmeshes_rescale_anisotropic(setup_gridmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
    config = copy.deepcopy(config)

    config['gridmeshes']['rescale-before-write'] = [2,3,4]
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    execution_dir, _workflow = launch_flow(template_dir, 1)

    # Note xyz vs zyx
    scaled_boxes = { label: box * [4,3,2] for label, box in object_boxes.items() }
    check_outputs(execution_dir, scaled_boxes, atol=[8, 6, 4])


def test_gridmeshes_skip_existing(setup_gridmeshes_config, disable_auto_retry):
    template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
    config = copy.deepcopy(config)

    config['gridmeshes']['skip-existing'] = True
    YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

    # Create an empty file for mesh 2
    os.makedirs(f"{template_dir}/meshes")
    open(f"{template_dir}/meshes/2.obj", 'wb').close()
    execution_dir, _workflow = launch_flow(template_dir, 1)

    # The file should have been left alone (still empty).
    assert open(f"{execution_dir}/meshes/2.obj", 'rb').read() == b''

    # But other meshes were generated.
    check_outputs(execution_dir, object_boxes, subset_labels=[1,3])


# def test_gridmeshes_from_body_source(setup_gridmeshes_config, disable_auto_retry):
#     template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
#     config = copy.deepcopy(config)
#     config["input"]["dvid"]["supervoxels"] = False
#     YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

#     execution_dir, _workflow = launch_flow(template_dir, 1)
#     #print(execution_dir)

#     # In this test, each supervoxel is its own body anyway.
#     check_outputs(execution_dir, object_boxes)

# def test_gridmeshes_from_body_source_subset_bodies(setup_gridmeshes_config, disable_auto_retry):
#     template_dir, config, _dvid_address, _repo_uuid, object_boxes, _object_sizes = setup_gridmeshes_config
#     config = copy.deepcopy(config)
#
#     config["input"]["dvid"].update({
#         "supervoxels": False
#     })

#     config["input"]["geometry"].update({
#         "message-block-shape": [128,128,128]
#     })

#     config["input"]["adapters"] = {
#         "rescale-level": 1
#     }

#     config["gridmeshes"].update({
#         "subset-bodies": [100,300],
#         "rescale-before-write": 2.0
#     })

#     YAML().dump(config, open(f"{template_dir}/workflow.yaml", 'w'))

#     execution_dir, _workflow = launch_flow(template_dir, 1)
#     #print(execution_dir)

#     # In this test, each supervoxel is its own body anyway.
#     check_outputs(execution_dir, object_boxes, [100,300])


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")

    import flyemflows
    os.chdir(os.path.dirname(flyemflows.__file__))
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_gridmeshes']
    args += ['-x']
    #args += ['-Werror']
    #args += ['-k', 'gridmeshes_basic']
    pytest.main(args)
