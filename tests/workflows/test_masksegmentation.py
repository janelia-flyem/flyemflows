import os
import shutil
import tempfile
import textwrap
from io import StringIO

import pytest
from ruamel.yaml import YAML
from requests import HTTPError

import h5py
import numpy as np
import pandas as pd

from neuclease.util import extract_subvol, ndindex_array, switch_cwd
from neuclease.dvid import create_labelmap_instance, fetch_labelmap_voxels, post_labelmap_voxels, post_merge, post_roi, fetch_labelindex, fetch_mapping
from neuclease.dvid.rle import runlength_encode_to_ranges

from flyemflows.util import upsample, downsample
from flyemflows.bin.launchflow import launch_flow
from flyemflows.bin.erase_from_labelindexes import erase_from_labelindexes
from neuclease.dvid.repo import create_instance
from neuclease.util.segmentation import BLOCK_STATS_DTYPES, block_stats_for_volume

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')

MAX_SCALE = 7

test_case_counter = 0

@pytest.fixture
def setup_dvid_segmentation_input(setup_dvid_repo, random_segmentation):
    dvid_address, repo_uuid = setup_dvid_repo
 
    # Since the same UUID is re-used for each test case,
    # this counter is a little hack used to make sure the segmentation
    # has a unique name each time, so that previous test cases don't
    # affect subsequent test casess.
    global test_case_counter
    test_case_counter += 1
 
    # Normally the MaskSegmentation workflow is used to update
    # a segmentation instance from a parent uuid to a child uuid.
    # But for this test, we'll simulate that by writing to two
    # different instances in the same uuid.
    input_segmentation_name = f'masksegmentation-input-{test_case_counter}'
    output_segmentation_name = f'masksegmentation-output-from-dvid-{test_case_counter}'

    # Agglomerate some supervoxels into bodies
    # Choose supervoxels that intersect three Z-planes at 64, 128, 192
    svs_1 = np.unique(random_segmentation[64])
    svs_2 = np.unique(random_segmentation[128])
    svs_3 = np.unique(random_segmentation[192])

    for instance in (input_segmentation_name, output_segmentation_name):
        create_labelmap_instance(dvid_address, repo_uuid, instance, max_scale=MAX_SCALE)

        # Start with an empty mapping (the repo/instance are re-used for each test case)
        post_labelmap_voxels(dvid_address, repo_uuid, instance, (0,0,0), random_segmentation, downres=True)
        post_merge(dvid_address, repo_uuid, instance, svs_1[0], svs_1[1:])
        post_merge(dvid_address, repo_uuid, instance, svs_2[0], svs_2[1:])
        post_merge(dvid_address, repo_uuid, instance, svs_3[0], svs_3[1:])

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
            # Choose a brick that doesn't cleanly divide into the bounding box
            message-block-shape: [192,64,64]
 
        output:
          dvid:
            server: {dvid_address}
            uuid: {repo_uuid}
            segmentation-name: {output_segmentation_name}
            supervoxels: true
            disable-indexing: true
 
        masksegmentation:
          mask-roi: {roi_name}
          batch-size: 5
          block-statistics-file: erased-block-statistics.h5
    """)
 
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        f.write(config_text)
 
    yaml = YAML()
    with StringIO(config_text) as f:
        config = yaml.load(f)
 
    return template_dir, config, random_segmentation, dvid_address, repo_uuid, roi_mask_s5, input_segmentation_name, output_segmentation_name


EXPORT_DEBUG_FILES = True
@pytest.mark.parametrize(
    'invert_mask,roi_dilation', [
        (False, 0),
        (True, 1) # FIXME: We don't yet verify voxel-accuracy of ROI dilation.
    ])
def test_masksegmentation_basic(setup_dvid_segmentation_input, invert_mask, roi_dilation, disable_auto_retry):
    template_dir, config, volume, dvid_address, repo_uuid, roi_mask_s5, input_segmentation_name, output_segmentation_name = setup_dvid_segmentation_input

    if invert_mask:
        roi_mask_s5 = ~roi_mask_s5

    config["masksegmentation"]["invert-mask"] = invert_mask
    config["masksegmentation"]["dilate-roi"] = roi_dilation

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
    output_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, output_box_zyx, scale=0, supervoxels=True)

    # Create a copy of the volume that contains only the voxels we removed
    erased_vol = volume.copy()
    erased_vol[~roi_mask] = 0

    if EXPORT_DEBUG_FILES:
        original_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, input_segmentation_name, output_box_zyx, scale=0, supervoxels=True)
        original_agglo_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, input_segmentation_name, output_box_zyx, scale=0)
        output_agglo_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, output_box_zyx, scale=0)
        np.save('/tmp/original-svs.npy', original_vol)
        np.save('/tmp/original-agglo.npy', original_agglo_vol)
        np.save('/tmp/output.npy', output_vol)
        np.save('/tmp/output-agglo.npy', output_agglo_vol)
        np.save('/tmp/expected.npy', expected_vol)
        np.save('/tmp/erased.npy', erased_vol)
        
        shutil.copyfile(f'{execution_dir}/roi-mask.h5', '/tmp/roi-mask.h5')
        if roi_dilation:
            shutil.copyfile(f'{execution_dir}/dilated-roi-mask.h5', '/tmp/dilated-roi-mask.h5')
        if invert_mask:
            shutil.copyfile(f'{execution_dir}/segmentation-mask.h5', '/tmp/segmentation-mask.h5')
        shutil.copyfile(f'{execution_dir}/final-mask.h5', '/tmp/final-mask.h5')

    if roi_dilation > 0:
        # FIXME: We don't yet verify voxel-accuracy of ROI dilation.
        return

    assert (output_vol == expected_vol).all(), \
        "Written vol does not match expected"

    scaled_expected_vol = expected_vol
    for scale in range(1, 1+MAX_SCALE):
        scaled_expected_vol = downsample(scaled_expected_vol, 2, 'labels-numba')
        scaled_output_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, output_box_zyx // 2**scale, scale=scale, supervoxels=True)

        if EXPORT_DEBUG_FILES:
            np.save(f'/tmp/expected-{scale}.npy', scaled_expected_vol)
            np.save(f'/tmp/expected-{scale}.npy', scaled_expected_vol)
            np.save(f'/tmp/output-{scale}.npy', scaled_output_vol)
        
        if scale <= 5:
            assert (scaled_output_vol == scaled_expected_vol).all(), \
                f"Written vol does not match expected at scale {scale}"
        else:
            # For scale 6 and 7, some blocks are not even changed,
            # but that means we would be comparing DVID's label
            # downsampling method to our method ('labels-numba').
            # The two don't necessarily give identical results in the case of 'ties',
            # so we'll just verify that the nonzero voxels match, at least.
            assert ((scaled_output_vol == 0) == (scaled_expected_vol == 0)).all(), \
                f"Written vol does not match expected at scale {scale}"
            

    block_stats_path = f'{execution_dir}/erased-block-statistics.h5'
    with h5py.File(block_stats_path, 'r') as f:
        stats_df = pd.DataFrame(f['stats'][:])
    
    #
    # Check the exported block statistics
    #
    stats_cols = [*BLOCK_STATS_DTYPES.keys()]
    assert stats_df.columns.tolist() == stats_cols
    stats_df = stats_df.sort_values(stats_cols).reset_index()
    
    expected_stats_df = block_stats_for_volume((64,64,64), erased_vol, input_box_zyx)
    expected_stats_df = expected_stats_df.sort_values(stats_cols).reset_index()

    assert len(stats_df) == len(expected_stats_df)
    assert (stats_df == expected_stats_df).all().all()

    #
    # Try updating the labelindexes
    #
    src_info = (dvid_address, repo_uuid, input_segmentation_name)
    dest_info = (dvid_address, repo_uuid, output_segmentation_name)
    with switch_cwd(execution_dir):
        erase_from_labelindexes(src_info, dest_info, block_stats_path, batch_size=10, threads=4)

    # Verify deleted supervoxels
    assert os.path.exists(f'{execution_dir}/deleted-supervoxels.csv')
    deleted_svs = set(pd.read_csv(f'{execution_dir}/deleted-supervoxels.csv')['sv'])

    orig_svs = {*pd.unique(volume.reshape(-1))} - {0}
    remaining_svs = {*pd.unique(expected_vol.reshape(-1))} - {0}
    expected_deleted_svs = orig_svs - remaining_svs
    assert deleted_svs == expected_deleted_svs

    # Verify remaining sizes
    expected_sv_counts = (pd.Series(expected_vol.reshape(-1), name='sv')
                            .value_counts()
                            .drop(0)
                            .sort_index()
                            .rename('count'))
    
    index_dfs = []
    for body in np.unique(fetch_mapping(*dest_info, remaining_svs)):
        index_df = fetch_labelindex(*dest_info, body, format='pandas').blocks
        index_dfs.append(index_df)
    
    sv_counts = (pd.concat(index_dfs, ignore_index=True)[['sv', 'count']]
                 .groupby('sv')['count']
                 .sum()
                 .sort_index())
    assert set(sv_counts.index.values) == set(expected_sv_counts.index.values)
    assert (sv_counts == expected_sv_counts).all(), \
        pd.DataFrame({'stored_count': sv_counts, 'expected_count': expected_sv_counts}).query('stored_count != expected_count')

    # Verify mapping
    # Deleted supervoxels exist in the mapping, but they map to 0.
    assert (fetch_mapping(*dest_info, [*deleted_svs]) == 0).all()
    
    # Remaining supervoxels still map to their original bodies
    assert (fetch_mapping(*dest_info, [*remaining_svs]) == fetch_mapping(*src_info, [*remaining_svs])).all()


def test_masksegmentation_resume(setup_dvid_segmentation_input, disable_auto_retry):
    template_dir, config, volume, dvid_address, repo_uuid, roi_mask_s5, _input_segmentation_name, output_segmentation_name = setup_dvid_segmentation_input

    brick_shape = config["input"]["geometry"]["message-block-shape"]
    batch_size = config["masksegmentation"]["batch-size"]
    
    # This is the total bricks in the volume, not necessarily
    # the total *processed* bricks, but it's close enough.
    total_bricks = np.ceil(np.prod(np.array(volume.shape) / brick_shape)).astype(int)
    total_batches = int(np.ceil(total_bricks / batch_size))

    # Skip over half of the original bricks.
    config["masksegmentation"]["resume-at"] = {
        "scale": 0,
        "batch-index": 1 + (total_batches // 2)
    }

    # re-dump config
    yaml = YAML()
    yaml.default_flow_style = False    
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    _execution_dir, workflow = launch_flow(template_dir, 1)
    final_config = workflow.config

    input_box_xyz = np.array( final_config['input']['geometry']['bounding-box'] )
    input_box_zyx = input_box_xyz[:,::-1]
    
    roi_mask = upsample(roi_mask_s5, 2**5)
    roi_mask = extract_subvol(roi_mask, input_box_zyx)
    
    masked_vol = extract_subvol(volume.copy(), input_box_zyx)
    masked_vol[roi_mask] = 0

    output_box_xyz = np.array( final_config['output']['geometry']['bounding-box'] )
    output_box_zyx = output_box_xyz[:,::-1]
    output_vol = fetch_labelmap_voxels(dvid_address, repo_uuid, output_segmentation_name, output_box_zyx, scale=0, supervoxels=True)

    #np.save('/tmp/original.npy', volume)
    #np.save('/tmp/output.npy', output_vol)

    # First part was untouched
    assert (output_vol[:128] == volume[:128]).all()

    # Last part was touched somewhere
    assert (output_vol[128:] != volume[128:]).any()


if __name__ == "__main__":
    if 'CLUSTER_TYPE' in os.environ:
        import warnings
        warnings.warn("Disregarding CLUSTER_TYPE when running via __main__")
    
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    args = ['-s', '--tb=native', '--pyargs', 'tests.workflows.test_masksegmentation']
    args += ['-x']
    #args += ['-Werror']
    #args += ['-k', 'masksegmentation_basic']
    #args += ['-k', 'masksegmentation_resume']
    pytest.main(args)
