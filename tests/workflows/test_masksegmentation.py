import os
import tempfile
import textwrap
from io import StringIO

import pytest
from ruamel.yaml import YAML
from requests import HTTPError

import h5py
import numpy as np
import pandas as pd

from neuclease.util import extract_subvol, ndindex_array
from neuclease.dvid import create_labelmap_instance, fetch_labelmap_voxels, post_labelmap_voxels, post_roi
from neuclease.dvid.rle import runlength_encode_to_ranges

from flyemflows.util import upsample, downsample
from flyemflows.bin.launchflow import launch_flow
from neuclease.dvid.repo import create_instance
from neuclease.util.segmentation import BLOCK_STATS_DTYPES, block_stats_for_volume

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')

MAX_SCALE = 7

@pytest.fixture
def setup_dvid_segmentation_input(setup_dvid_repo, random_segmentation):
    dvid_address, repo_uuid = setup_dvid_repo
 
    # Normally the MaskSegmentation workflow is used to update
    # a segmentation instance from a parent uuid to a child uuid.
    # But for this test, we'll simulate that by writing to two
    # different instances in the same uuid.
    input_segmentation_name = 'segmentation-input'
    output_segmentation_name = 'segmentation-output-from-dvid'

    for instance in (input_segmentation_name, output_segmentation_name):    
        try:
            create_labelmap_instance(dvid_address, repo_uuid, instance, max_scale=MAX_SCALE)
        except HTTPError as ex:
            if ex.response is not None and 'already exists' in ex.response.content.decode('utf-8'):
                pass
        
        post_labelmap_voxels(dvid_address, repo_uuid, instance, (0,0,0), random_segmentation, downres=True)

    # Create an ROI to test with -- a sphere with scale-5 resolution
    shape_s5 = np.array(random_segmentation.shape) // 2**5
    midpoint_s5 = shape_s5 / 2
    radius = midpoint_s5.min()
    
    coords_s5 = ndindex_array(*shape_s5)
    distances = np.sqrt(np.sum((coords_s5 - midpoint_s5)**2, axis=1))
    keep = (distances < radius)
    coords_s5 = coords_s5[keep, :]
    
    roi_ranges = runlength_encode_to_ranges(coords_s5)
    roi_name = 'masksegmentation-test-roi'

    try:
        create_instance(dvid_address, repo_uuid, roi_name, 'roi')
    except HTTPError as ex:
        if ex.response is not None and 'already exists' in ex.response.content.decode('utf-8'):
            pass

    post_roi(dvid_address, repo_uuid, roi_name, roi_ranges)
    
    roi_mask_s5 = np.zeros(shape_s5, dtype=bool)
    roi_mask_s5[(*coords_s5.transpose(),)] = True

    template_dir = tempfile.mkdtemp(suffix="masksegmentation-from-dvid-template")
 
    config_text = textwrap.dedent(f"""\
        workflow-name: masksegmentation
        cluster-type: {CLUSTER_TYPE}
         
        input:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {input_segmentation_name}
            supervoxels: true
           
          geometry:
            message-block-shape: [128,64,64]
 
        output:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {output_segmentation_name}
            supervoxels: true
            disable-indexing: true
 
        masksegmentation:
          mask-roi: {roi_name}
          batch-size: 10
          block-statistics-file: erased-block-statistics.h5
    """)
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)
 
    return template_dir, config, random_segmentation, dvid_address, repo_uuid, roi_mask_s5, output_segmentation_name

@pytest.mark.parametrize('invert_mask', [True, False])
def test_masksegmentation(setup_dvid_segmentation_input, invert_mask, disable_auto_retry):
    template_dir, config, volume, dvid_address, repo_uuid, roi_mask_s5, output_segmentation_name = setup_dvid_segmentation_input

    if invert_mask:
        roi_mask_s5 = ~roi_mask_s5

    config["masksegmentation"]["invert-mask"] = invert_mask

    # re-dump config
    yaml = YAML()
    yaml.default_flow_style = False    
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)
    
    execution_dir, workflow = launch_flow(template_dir, 1)
    final_config = workflow.config

    input_box_xyz = np.array( final_config['input']['geometry']['bounding-box'] )
    input_box_zyx = input_box_xyz[:,::-1]
    
    roi_mask = upsample(roi_mask_s5, 2**5)
    roi_mask = extract_subvol(roi_mask, input_box_zyx)
    
    expected_vol = extract_subvol(volume.copy(), input_box_zyx)
    expected_vol[roi_mask] = 0
    
    output_box_xyz = np.array( final_config['output']['geometry']['bounding-box'] )
    output_box_zyx = output_box_xyz[:,::-1]
    output_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, output_box_zyx, scale=0)

    # Create a copy of the volume that contains only the voxels we removed
    erased_vol = volume.copy()
    erased_vol[~roi_mask] = 0

    # Debug visualization
    #np.save('/tmp/erased.npy', erased_vol)
    #np.save('/tmp/output.npy', output_vol)
    #np.save('/tmp/expected.npy', expected_vol)

    assert (output_vol == expected_vol).all(), \
        "Written vol does not match expected"

    for scale in range(1, 1+MAX_SCALE):
        expected_vol = downsample(expected_vol, 2, 'labels-numba')
        output_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, output_box_zyx // 2**scale, scale=scale)

        np.save(f'/tmp/output-{scale}.npy', output_vol)
        np.save(f'/tmp/expected-{scale}.npy', expected_vol)
        
        if scale <= 5:
            assert (output_vol == expected_vol).all(), \
                f"Written vol does not match expected at scale {scale}"
        else:
            # For scale 6 and 7, some blocks are not even changed,
            # but that means we would be comparing DVID's label
            # downsampling method to our method ('labels-numba').
            # The two don't necessarily give identical results in the case of 'ties',
            # so we'll just verify that the nonzero voxels match, at least.
            assert ((output_vol == 0) == (expected_vol == 0)).all(), \
                f"Written vol does not match expected at scale {scale}"
            
        
    with h5py.File(f'{execution_dir}/erased-block-statistics.h5', 'r') as f:
        stats_df = pd.DataFrame(f['stats'][:])
    
    # Check the exported block statistics
    stats_cols = [*BLOCK_STATS_DTYPES.keys()]
    assert stats_df.columns.tolist() == stats_cols
    stats_df = stats_df.sort_values(stats_cols).reset_index()
    
    expected_stats_df = block_stats_for_volume((64,64,64), erased_vol, input_box_zyx)
    expected_stats_df = expected_stats_df.sort_values(stats_cols).reset_index()

    assert len(stats_df) == len(expected_stats_df)
    assert (stats_df == expected_stats_df).all().all()


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")
    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_masksegmentation']
    args += ['-x']
    pytest.main(args)
