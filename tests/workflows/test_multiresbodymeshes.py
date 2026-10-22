import os
import tempfile
import textwrap
from io import StringIO

import numpy as np
import pandas as pd
from requests import HTTPError
from skimage.util import view_as_blocks

from neuclease.util import ndrange
from neuclease.dvid import create_labelmap_instance, post_labelarray_blocks, fetch_key, fetch_keyrangevalues
from vol2mesh.multires import parse_manifest

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchflow import launch_flow

from .test_chunkedbodymeshes import create_test_object

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')


@pytest.fixture
def setup_dvid_segmentation_input(setup_dvid_repo):
    dvid_address, repo_uuid = setup_dvid_repo
    input_segmentation_name = 'segmentation-multiresbodymeshes-input'

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

    template_dir = tempfile.mkdtemp(suffix="multiresbodymeshes-template")

    config_text = textwrap.dedent(f"""\
        workflow-name: multiresbodymeshes
        cluster-type: {CLUSTER_TYPE}

        multiresbodymeshes:
            dvid:
                server: {dvid_address}
                uuid: {repo_uuid}
                segmentation-name: {input_segmentation_name}

            mesh-config:
                # It not normal for uuid to match base-uuid, but it should be okay for this simple test.
                base-uuid: {repo_uuid}
                chunk-shape-s0: [128, 128, 128]
                chunk-halo: 2
                source-scale: 1
                smoothing: 3
                lod-error-thresholds: [4.0, 40.0]

            force-update: true

            bodies: [100, 200] # 200 doesn't exist, will trigger deletion

            # Synchronous test.
            chunk-processes: 0
    """)

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)

    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)

    return template_dir, config, dvid_address, repo_uuid, input_segmentation_name


def test_multiresbodymeshes(setup_dvid_segmentation_input):
    template_dir, _config, dvid_address, repo_uuid, input_segmentation_name = setup_dvid_segmentation_input

    _execution_dir, _workflow = launch_flow(template_dir, 1)

    seg = input_segmentation_name
    mesh_info = fetch_keyrangevalues(dvid_address, repo_uuid, f"{seg}_multires_mesh_info", as_json=True)
    mesh_info_df = pd.DataFrame(mesh_info.values()).set_index('body')
    assert mesh_info_df.loc[100, 'method'] == 'multires-chunks'
    assert mesh_info_df.loc[100, 'num-lods'] == 2
    assert mesh_info_df.loc[200, 'method'] == 'deleted'

    index = fetch_key(dvid_address, repo_uuid, f"{seg}_multires_meshes", "100.index")
    assert parse_manifest(index)['num_lods'] == 2
    fetch_key(dvid_address, repo_uuid, f"{seg}_multires_meshes", "100")
    fetch_key(dvid_address, repo_uuid, f"{seg}_multires_meshes", "100.chunks")

    with pytest.raises(HTTPError):
        fetch_key(dvid_address, repo_uuid, f"{seg}_multires_meshes", "200")


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")

    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_multiresbodymeshes'])
