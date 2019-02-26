import os
import tempfile
import textwrap
from io import StringIO

import h5py
import pytest
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from flyemflows.util import upsample
from flyemflows.bin.launchflow import launch_flow
from neuclease.util import extract_subvol, BLOCK_STATS_DTYPES
from neuclease.dvid import create_labelmap_instance, post_labelmap_voxels

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')

@pytest.fixture
def setup_sparseblockstats(setup_dvid_repo):
    dvid_address, repo_uuid = setup_dvid_repo
    
    _ = 0
    volume_layout = [[_,_,_,_ ,_,_,_,_],
                     [_,_,_,_ ,_,4,_,_],
                     [_,1,1,1 ,_,_,1,1],
                     [_,1,_,_ ,_,_,_,_],

                     [_,1,_,_ ,_,_,_,_],
                     [_,1,_,2 ,2,2,2,_],
                     [_,_,_,_ ,_,_,_,_],
                     [_,3,_,_ ,_,3,_,1]]
    
    lowres_volume = np.zeros((4,8,8), np.uint64)
    lowres_volume[:] = volume_layout

    volume = upsample(lowres_volume, 16)
    assert volume.shape == (64,128,128)

    input_segmentation_name = 'sparseblockstats-input'
    create_labelmap_instance(dvid_address, repo_uuid, input_segmentation_name)
    post_labelmap_voxels(dvid_address, repo_uuid, input_segmentation_name, (0,0,0), volume)

    # Mask is same as input, but times 10
    mask_volume = volume*10
    mask_segmentation_name = 'sparseblockstats-mask'
    create_labelmap_instance(dvid_address, repo_uuid, mask_segmentation_name)
    post_labelmap_voxels(dvid_address, repo_uuid, mask_segmentation_name, (0,0,0), mask_volume)


    config_text = textwrap.dedent(f"""\
        workflow-name: sparseblockstats
        cluster-type: {CLUSTER_TYPE}
         
        input:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {input_segmentation_name}
            supervoxels: true
           
          geometry:
            message-block-shape: [64,64,64]
            bounding-box: [[0,0,0], [128,128,64]]
 
        mask-input:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {mask_segmentation_name}
            supervoxels: true
           
          geometry:
            message-block-shape: [64,64,64]
            bounding-box: [[0,0,0], [128,128,64]]
 
        sparseblockstats:
          mask-labels: [20,40] # Avoids the top-left block
    """)
 
    template_dir = tempfile.mkdtemp(suffix="sparseblockstats-template")

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)

    yaml = YAML()
    yaml.default_flow_style = False
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    return template_dir, config, volume, mask_volume, dvid_address, repo_uuid


def test_sparseblocksstats(setup_sparseblockstats, disable_auto_retry):
    template_dir, _config, input_volume, _mask_volume, _dvid_address, _repo_uuid = setup_sparseblockstats
    
    execution_dir, workflow = launch_flow(template_dir, 1)
    _final_config = workflow.config
    
    with h5py.File(f'{execution_dir}/block-statistics.h5', 'r') as f:
        assert f['stats'].dtype == np.dtype(list(BLOCK_STATS_DTYPES.items()))
        stats_df = pd.DataFrame(f['stats'][:])

    for row in stats_df.itertuples():
        corner = np.array((row.z, row.y, row.x))
        block_box = np.array([corner, corner+64])
        block = extract_subvol(input_volume, block_box)
        assert (block == row.segment_id).sum() == row.count

    assert len(stats_df.query('z == 0 and y == 0 and x == 0')) == 0, \
        "Was not supposed to compute stats for the first block!"

    block_coords = stats_df[['z', 'y', 'x']].sort_values(['z', 'y', 'x']).drop_duplicates(['z', 'y', 'x']).values
    assert (block_coords == 64*np.array([[0,0,1], [0,1,0], [0,1,1]])).all(), \
        "Did not cover blocks for the selected labels!"
    
    for row in stats_df.itertuples():
        corner = np.array((row.z, row.y, row.x))
        block_box = np.array([corner, corner+64])
        block = extract_subvol(input_volume, block_box)
        assert (block == row.segment_id).sum() == row.count


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")
    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_sparseblockstats']
    #args = ['-k', 'sparseblocksstats'] + args
    pytest.main(args)
