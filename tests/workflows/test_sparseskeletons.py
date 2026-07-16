import os
import tempfile
import textwrap
from io import StringIO

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from neuclease.util import ndrange
from neuclease.dvid import create_labelmap_instance, post_labelarray_blocks
from neuclease.util.skeleton import swc_to_dataframe

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchflow import launch_flow

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')


def create_test_object():
    # Create a test object (shaped like an 'X')
    center_line_img = np.zeros((128, 128, 128), dtype=np.uint32)
    for i in range(128):
        center_line_img[i, i, i] = 1
        center_line_img[127 - i, i, i] = 1

    # Scipy distance_transform_edt conventions are opposite of vigra:
    # it calculates distances of non-zero pixels to the zero pixels.
    center_line_img = 1 - center_line_img
    distance_to_line = distance_transform_edt(center_line_img)
    binary_vol = (distance_to_line <= 10).astype(np.uint8)
    return binary_vol


@pytest.fixture
def setup_dvid_segmentation_input(setup_dvid_repo):
    dvid_address, repo_uuid = setup_dvid_repo
    input_segmentation_name = 'segmentation-sparseskeletons-input'

    create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name, max_scale=3)

    label_vol = create_test_object().astype(np.uint64)
    label_vol *= 100

    corners = np.array(list(ndrange((0, 0, 0), label_vol.shape, (64, 64, 64))))

    from skimage.util import view_as_blocks
    blockwise_vol = view_as_blocks(label_vol, (64, 64, 64))
    blocks = blockwise_vol.reshape(-1, 64, 64, 64)

    # post to dvid
    post_labelarray_blocks(dvid_address, repo_uuid, input_segmentation_name, corners, blocks, downres=True, noindexing=False)

    template_dir = tempfile.mkdtemp(suffix="sparseskeletons-template")

    config_text = textwrap.dedent(f"""\
        workflow-name: sparseskeletons
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

        sparseskeletons:
          scale: 1
          block-shape: [64, 64, 64]
          halo: 8
          closing-radius: 0
          format: swc
          bodies: [100, 200] # 200 doesn't exist, will fail (see below)

        output:
          directory: skeletons
    """)

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)

    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)

    return template_dir, config, dvid_address, repo_uuid


def test_sparseskeletons(setup_dvid_segmentation_input):
    template_dir, _config, _dvid_address, _repo_uuid = setup_dvid_segmentation_input

    execution_dir, _workflow = launch_flow(template_dir, 1)

    # Body 100 exists -> an SWC file should be written.
    swc_path = f"{execution_dir}/skeletons/100.swc"
    assert os.path.exists(swc_path)

    swc_text = open(swc_path).read()
    # Header includes the NeuTu-compatible comment keys.
    assert '"dataName": "segmentation-sparseskeletons-input"' in swc_text
    assert '"mutation id":' in swc_text

    # The SWC body parses into a non-trivial skeleton.
    sk_df = swc_to_dataframe(swc_text)
    assert len(sk_df) > 0

    df = pd.read_feather(f'{execution_dir}/skeleton-stats.feather')
    assert len(df) == 2
    df = df.set_index('body')
    assert df.loc[100, 'status'] == 'success'

    # The second body didn't exist, so it fails (but doesn't kill the workflow).
    assert df.loc[200, 'status'] == 'failed-download'
    assert not os.path.exists(f"{execution_dir}/skeletons/200.swc")


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")

    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_sparseskeletons'])
