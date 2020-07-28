import os
import tempfile
import textwrap
from io import StringIO

import pytest
from ruamel.yaml import YAML

import h5py
import numpy as np

from flyemflows.bin.launchflow import launch_flow

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')


##
## TODO: This doesn't exercise the ROI masking.
##       Would need dvid test data and an ROI for that.
##

@pytest.fixture
def setup_hdf5_inputs():
    template_dir = tempfile.mkdtemp(suffix="test-maskedcopy")

    input_vol = np.zeros((256, 256, 256), dtype=np.uint64)
    input_vol[0:100] = 1
    input_vol[100:200] = 2
    input_vol[200:] = 2

    mask_vol = np.zeros((256, 256, 256), dtype=np.uint64)
    mask_vol[150:200, 150:200, 150:200] = 1

    input_path = f"{template_dir}/input-vol.h5"
    mask_path = f"{template_dir}/mask-vol.h5"
    output_path = f"{template_dir}/output-vol.h5"

    with h5py.File(input_path, 'w') as f:
        f['volume'] = input_vol

    with h5py.File(mask_path, 'w') as f:
        f['volume'] = mask_vol

    config_text = textwrap.dedent(f"""\
        workflow-name: maskedcopy
        cluster-type: {CLUSTER_TYPE}

        input:
          hdf5:
            path: {input_path}
            dataset: volume

          geometry:
            message-block-shape: [256,64,64]

        mask:
          hdf5:
            path: {mask_path}
            dataset: volume

          geometry:
            message-block-shape: [256,64,64]

        output:
          hdf5:
            path: {output_path}
            dataset: volume
            writable: true
            dtype: uint64

          geometry:
            message-block-shape: [256,64,64]
            bounding-box: [[0,0,0], [256, 256, 256]]

        maskedcopy:
          batch-size: 4
    """)

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)

    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)

    return template_dir, config, input_vol, mask_vol, output_path


def test_maskedcopy(setup_hdf5_inputs):
    template_dir, config, input_vol, mask_vol, output_path = setup_hdf5_inputs
    expected_vol = np.where(mask_vol, input_vol, 0)

    execution_dir, _workflow = launch_flow(template_dir, 1)

    with h5py.File(output_path, 'r') as f:
        output_vol = f['volume'][:]

    assert (output_vol == expected_vol).all()


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")

    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_maskedcopy']
    # args = ['-k', 'maskedcopy'] + args
    pytest.main(args)
