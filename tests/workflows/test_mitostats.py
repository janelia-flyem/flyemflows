import os
import pickle
import tempfile
import textwrap
from io import StringIO
from functools import lru_cache

import pytest
from ruamel.yaml import YAML

import h5py
import numpy as np
import pandas as pd
from numba import njit

from neuclease.util import box_intersection, box_to_slicing as b2s
from flyemflows.bin.launchflow import launch_flow

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')


##
## TODO: These tests don't exercise the ROI masking.
##       Would need dvid test data and an ROI for that.
##

@njit
def add_sphere(vol, center, radius, label):
    """
    Add a spherical segment to the given volume,
    with the given center, radius, and label value.
    Modifies vol in-place.

    The sphere need not be completely contained within the volume.
    If the center is placed near the edge of the volume (or even
    outside of it), the sphere will merely be cropped.
    """
    Z, Y, X = vol.shape
    cz, cy, cx = center

    z0 = max(0, cz - radius)
    y0 = max(0, cy - radius)
    x0 = max(0, cx - radius)

    z1 = min(Z, cz + radius + 1)
    y1 = min(Y, cy + radius + 1)
    x1 = min(X, cx + radius + 1)

    for z in range(z0,z1):
        for y in range(y0,y1):
            for x in range(x0,x1):
                d = np.sqrt((z-cz)**2 + (y-cy)**2 + (x-cx)**2)
                if d <= radius:
                    vol[z,y,x] = label

@pytest.fixture
def setup_hdf5_inputs():
    template_dir = tempfile.mkdtemp(suffix="test-mitostats")

    seg_vol = np.zeros((256, 256, 256), dtype=np.uint64)
    add_sphere(seg_vol, (32,32,32), 8, 1)
    add_sphere(seg_vol, (128, 128, 128), 32, 2)

    # Give mito 2 a class of 2 in the center slice (Z=128),
    # but symmetric wings of class 3 and 4.
    mask_vol = seg_vol.copy()
    view = mask_vol[64:128]
    view[(view != 0)] = 3
    view = mask_vol[129:192]
    view[(view != 0)] = 4

    seg_path = f"{template_dir}/seg-vol.h5"
    mask_path = f"{template_dir}/mask-vol.h5"
    output_path = f"{template_dir}/output-vol.h5"

    with h5py.File(seg_path, 'w') as f:
        f['volume'] = seg_vol

    with h5py.File(mask_path, 'w') as f:
        f['volume'] = mask_vol

    config_text = textwrap.dedent(f"""\
        workflow-name: mitostats
        cluster-type: {CLUSTER_TYPE}

        mito-seg:
          hdf5:
            path: {seg_path}
            dataset: volume

          geometry:
            message-block-shape: [256,64,64]

        mito-masks:
          hdf5:
            path: {mask_path}
            dataset: volume

          geometry:
            message-block-shape: [256,64,64]

        mitostats:
          min-size: 100
    """)

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)

    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)

    return template_dir, config, seg_vol, mask_vol, output_path


def test_mitostats(setup_hdf5_inputs):
    template_dir, config, seg_vol, mask_vol, output_path = setup_hdf5_inputs
    execution_dir, _workflow = launch_flow(template_dir, 1)

    stats_df = pickle.load(open(f'{execution_dir}/stats_df.pkl', 'rb'))

    assert (stats_df.loc[1, ['z', 'y', 'x']] == [32,32,32]).all()
    assert (stats_df.loc[2, ['z', 'y', 'x']] == [128,128,128]).all()

    # These classes should be symmetric. See above.
    assert (stats_df.loc[2, 'class_3'] == stats_df.loc[2, 'class_4'])

    assert stats_df.loc[1, 'total_size'] == (seg_vol == 1).sum()
    assert stats_df.loc[2, 'total_size'] == (seg_vol == 2).sum()


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")

    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_mitostats']
    # args = ['-k', 'mitostats'] + args
    pytest.main(args)
