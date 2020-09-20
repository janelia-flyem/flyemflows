import os
import copy
import pickle
import logging
from textwrap import dedent

import vigra
import numpy as np
import pandas as pd
import dask.bag as db
from skimage.measure import label
from skimage.morphology import binary_dilation, ball

from dvidutils import LabelMapper
from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import (Timer, round_box, SparseBlockMask, boxes_from_grid, iter_batches, tqdm_proxy,
                            ndindex_array, contingency_table, edge_mask, binary_edge_mask, mask_for_labels,
                            approximate_hulls_for_segments, apply_mask_for_labels, box_to_slicing)
from neuclease.dvid import fetch_labelmap_voxels, fetch_roi, fetch_sizes

from ..util import replace_default_entries
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService, ScaledVolumeService
from . import Workflow

logger = logging.getLogger(__name__)


class MitoRepair(Workflow):
    pass


MITO_VOL_FRAC = 0.4
MITO_EDGE_FRAC = 0.4


def mito_body_assignments_for_box(body_seg_svc, mito_class_svc, central_box_s0, halo_s0=128, scale=1,
                                  body_seg_dvid_src=None, viewer=None, res0=8, hull_scale=0):
    """
    Identify small bodies in the segmentation for whom a
    significant fraction are covered by the mito mask.
    For each of those "mito bodies" (which may contain both mito and
    non-mito voxels as far as the mask is concerned),
    determine which of the large non-mito neighboring bodies
    (if any) is the best candidate to merge the body into.
    The results are returned in a DataFrame which also contains basic
    stats about the size of each body, including its size within the
    "central" box vs. "halo" that was also used when this block was processed.
    """
    hull_scale = max(hull_scale, scale)
    res = (2**scale) * res0

    central_box_s0 = np.asarray(central_box_s0)
    box_s0 = np.array([central_box_s0[0] - halo_s0,
                       central_box_s0[1] + halo_s0])

    box = box_s0 // (2**scale)
    halo = halo_s0 // (2**scale)

    with Timer("Fetching neuron segmentation", logger):
        body_seg = body_seg_svc.get_subvolume(box, scale)
        update_seg_layer(viewer, f'body_seg_{res}nm', body_seg, scale, box, res0)

    with Timer("Fetching mito class segmentation", logger):
        mito_seg = mito_class_svc.get_subvolume(box, scale)
        # Relabel 4 -> 0 and cast to uint8
        mito_seg = np.array([0,1,2,3,0], np.uint8)[mito_seg]
        mito_binary = (mito_seg != 0)
        update_seg_layer(viewer, f'mito_mask_{res}nm', mito_seg, scale, box, res0)

    with Timer("Identifying mostly-mito bodies", logger):
        mito_bodies, mito_bodies_mask, mito_body_ct = identify_mito_bodies(body_seg, mito_binary, box, scale, halo, body_seg_dvid_src, viewer, res0)

    with Timer("Computing hull seeds", logger):
        hull_seeds_df, seed_bodies, hull_seed_mask, hull_seeds_cc = compute_hull_seeds(mito_bodies_mask, mito_binary, body_seg, box, scale, viewer, res0)

    with Timer("Computing hulls", logger):
        hull_masks = approximate_hulls_for_segments(hull_seeds_cc, 2**(hull_scale-scale), as_masks=True)
        if viewer:
            # Populate the hull_seg with hulls from smallest to largest,
            # so larger hulls overwrite smaller hulls.
            # (This is more representative of how the masks are prioritized below.)
            hull_seg = np.zeros_like(body_seg)
            for cc, (mask_box, mask) in sorted(hull_masks.items(), key=lambda c_bm: c_bm[1][1].sum()):
                hull_seg_view = hull_seg[box_to_slicing(*mask_box)]
                hull_seg_view[:] = np.where(mask, seed_bodies[cc], hull_seg_view)
            update_seg_layer(viewer, 'initial-hull-seg', hull_seg, scale, box, res0)

    with Timer("Selecting mito merges", logger):
        final_table = select_hulls_for_mito_bodies(mito_body_ct, mito_bodies_mask, mito_binary, body_seg,
                                                   hull_masks, seed_bodies, box, scale, viewer, res0, progress=False)

    return final_table


def identify_mito_bodies(body_seg, mito_binary, box, scale, halo, body_seg_dvid_src=None, viewer=None, res0=8):
    # Identify segments that are mostly mito
    ct = contingency_table(body_seg, mito_binary).reset_index().rename(columns={'left': 'body', 'right': 'is_mito'})
    ct = ct.pivot(index='body', columns='is_mito', values='voxel_count').fillna(0).rename(columns={0: 'non_mito', 1: 'mito'})
    ct[['mito', 'non_mito']] *= ((2**scale)**3)

    ct['body_size_local'] = ct.eval('mito+non_mito')
    ct['mito_frac_local'] = ct.eval('mito/body_size_local')
    ct = ct.sort_values('mito_frac_local', ascending=False)

    # Also compute the halo vs. non-halo sizes of every body.
    central_box = (box - box[0]) + [[halo, halo, halo], [-halo, -halo, -halo]]
    central_body_seg = body_seg[box_to_slicing(*central_box)]
    central_sizes = (pd.Series(central_body_seg.ravel('K'))
                    .value_counts()
                    .rename('body_size_central')
                    .rename_axis('body'))

    central_mask = np.ones(central_box[1] - central_box[0], bool)
    update_mask_layer(viewer, 'central-box', central_mask, scale, central_box + box[0])

    ct = ct.merge(central_sizes, 'left', on='body').fillna(0)
    ct['halo_size'] = ct.eval('body_size_local - body_size_central')
    ct = ct.query('body != 0')

    # Immediately drop bodies that reside only in the halo
    ct = ct.query('body_size_central > 0').copy()

    # For the bodies that MIGHT pass the mito threshold (based on their local size)
    # fetch their global size, if a dvid source was provided.
    # If not, we'll just use the local size, which is less accurate but
    # faster since we've already got it.
    if body_seg_dvid_src is not None:
        local_mito_bodies = ct.query('mito_frac_local >= @MITO_EDGE_FRAC').index
        body_sizes = fetch_sizes(*body_seg_dvid_src, local_mito_bodies).rename('body_sizes')
        ct = ct.merge(body_sizes, 'left', on='body')
    else:
        ct['body_size'] = ct['body_size_local']

    # Due to downsampling effects, bodies can be larger at scale-1 than at scale-0, especially for tiny volumes.
    ct['mito_frac_global_vol'] = np.minimum(ct.eval('mito/body_size'), 1.0)

    # Calculate the proportion of mito edge pixels
    body_edges = np.where(edge_mask(body_seg, 'both'), body_seg, np.uint64(0))
    edge_ct = contingency_table(body_edges, mito_binary).reset_index().rename(columns={'left': 'body', 'right': 'is_mito'})
    edge_ct = edge_ct.pivot(index='body', columns='is_mito', values='voxel_count').fillna(0).rename(columns={0: 'non_mito', 1: 'mito'})

    # Surface area scales with square of resolution, not cube
    edge_ct[['mito', 'non_mito']] *= ((2**scale)**2)

    edge_ct['body_size_local'] = edge_ct.eval('mito+non_mito')
    edge_ct['mito_frac_local'] = edge_ct.eval('mito/body_size_local')
    edge_ct = edge_ct.sort_values('mito_frac_local', ascending=False)
    edge_ct = edge_ct.query('body != 0')

    full_ct = ct.merge(edge_ct, 'inner', on='body', suffixes=['_vol', '_edge'])
    filtered_ct = full_ct.query('mito_frac_global_vol >= @MITO_VOL_FRAC and mito_frac_local_edge >= @MITO_EDGE_FRAC')

    mito_bodies = filtered_ct.index
    mito_bodies_mask = mask_for_labels(body_seg, mito_bodies)
    update_mask_layer(viewer, 'mito-bodies-mask', mito_bodies_mask, scale, box, res0)

    filtered_ct = filtered_ct.copy()
    return mito_bodies, mito_bodies_mask, filtered_ct


def compute_hull_seeds(mito_bodies_mask, mito_binary, body_seg, box, scale, viewer=None, res0=8):
    # Select the voxels in the "mito bodies" that overlay the mito mask
    hull_seed_mask = mito_bodies_mask.copy()
    hull_seed_mask[~mito_binary] = 0

    # Dilate by 2 to find the adjacent bodies
    hull_seed_mask = binary_dilation(hull_seed_mask, ball(2))

    # But only keep the dilated voxels belonging to large (non-mito) bodies
    hull_seed_mask[mito_bodies_mask] = 0

    # update_seg_layer(v, f'hull-seed-mask', hull_seed_mask.astype(np.uint64), SCALE, box)

    hull_seeds = np.where(hull_seed_mask, body_seg, np.uint64(0))
    update_seg_layer(viewer, 'hull-seeds', hull_seeds, scale, box)

    hull_seeds_cc = label(hull_seeds, connectivity=3).view(np.uint64)
    hull_seeds_df = (pd.DataFrame({'hull_seed_body': hull_seeds[hull_seeds != 0],
                                   'hull_seed_cc': hull_seeds_cc[hull_seeds != 0]})
                        .groupby(['hull_seed_cc'])['hull_seed_body'].agg(['first', 'count'])
                        .sort_index())

    hull_seeds_df.columns = ['body', 'seed_size']

    # seed_bodies maps from hull_cc back to the body it came from.
    assert {*hull_seeds_df.index} == {*range(1, 1+len(hull_seeds_df))}
    seed_bodies = np.array([0] + [*hull_seeds_df['body']], dtype=np.uint64)

    return hull_seeds_df, seed_bodies, hull_seed_mask, hull_seeds_cc


def select_hulls_for_mito_bodies(mito_body_ct, mito_bodies_mask, mito_binary, body_seg, hull_masks,
                                 seed_bodies, box, scale, viewer=None, res0=8, progress=False):

    mito_bodies_mito_seg = np.where(mito_bodies_mask & mito_binary, body_seg, 0)
    nonmito_body_seg = np.where(mito_bodies_mask, 0, body_seg)

    hull_cc_overlap_stats = []
    for hull_cc, (mask_box, mask) in tqdm_proxy(hull_masks.items(), disable=not progress):
        mbms = mito_bodies_mito_seg[box_to_slicing(*mask_box)]
        masked_hull_cc_bodies = np.where(mask, mbms, 0)
        # Faster to check for any non-zero values at all before trying to count them.
        # This early check saves a lot of time in practice.
        if not masked_hull_cc_bodies.any():
            continue

        # This hull was generated from a particular seed body (non-mito body).
        # If it accidentally overlaps with any other non-mito bodies,
        # then delete those voxels from the hull.
        # If that causes the hull to become split apart into multiple connected components,
        # then keep only the component(s) which overlap the seed body.
        seed_body = seed_bodies[hull_cc]
        nmbs = nonmito_body_seg[box_to_slicing(*mask_box)]
        other_bodies = set(pd.unique(nmbs[mask])) - {0, seed_body}
        if other_bodies:
            # Keep only the voxels on mito bodies or on the
            # particular non-mito body for this hull (the "seed body").
            mbm = mito_bodies_mask[box_to_slicing(*mask_box)]
            mask[:] &= (mbm | (nmbs == seed_body))
            mask = vigra.taggedView(mask, 'zyx')
            mask_cc = vigra.analysis.labelMultiArrayWithBackground(mask.view(np.uint8))
            if mask_cc.max() > 1:
                mask_ct = contingency_table(mask_cc, nmbs).reset_index()
                keep_ccs = mask_ct['left'].loc[(mask_ct['left'] != 0) & (mask_ct['right'] == seed_body)]
                mask[:] = mask_for_labels(mask_cc, keep_ccs)

        mito_bodies, counts = np.unique(masked_hull_cc_bodies, return_counts=True)
        overlaps = pd.DataFrame({'mito_body': mito_bodies,
                                 'overlap': counts,
                                 'hull_cc': hull_cc,
                                 'hull_size': mask.sum(),
                                 'hull_body': seed_body})
        hull_cc_overlap_stats.append(overlaps)

    if len(hull_cc_overlap_stats) == 0:
        logger.warning("Could not find any matches for any mito bodies!")
        mito_body_ct['hull_body'] = np.nan
        return mito_body_ct

    hull_cc_overlap_stats = pd.concat(hull_cc_overlap_stats, ignore_index=True)
    hull_cc_overlap_stats = hull_cc_overlap_stats.query('mito_body != 0').copy()

    # Aggregate the stats for each body and the hull bodies it overlaps with,
    # Select the hull_body with the most overlap, or in the case of ties, the hull body that is largest overall.
    # (Ties are probably more common in the event that two hulls completely encompass a small mito body.)
    hull_body_overlap_stats = hull_cc_overlap_stats.groupby(['mito_body', 'hull_body'])[['overlap', 'hull_size']].sum()
    hull_body_overlap_stats = hull_body_overlap_stats.sort_values(['mito_body', 'overlap', 'hull_size'], ascending=False)
    hull_body_overlap_stats = hull_body_overlap_stats.reset_index()

    mito_hull_selections = (hull_body_overlap_stats.drop_duplicates('mito_body').set_index('mito_body')['hull_body'])
    mito_body_ct = mito_body_ct.merge(mito_hull_selections, 'left', left_index=True, right_index=True)

    if viewer:
        assert mito_hull_selections.index.dtype == mito_hull_selections.values.dtype == np.uint64
        mito_hull_mapper = LabelMapper(mito_hull_selections.index.values, mito_hull_selections.values)
        remapped_body_seg = mito_hull_mapper.apply(body_seg, True)
        remapped_body_seg = apply_mask_for_labels(remapped_body_seg, mito_hull_selections.values)
        update_seg_layer(viewer, 'altered-bodies', remapped_body_seg, scale, box)

        # Show the final hull masks (after erasure of non-target bodies)
        assert sorted(hull_masks.keys()) == [*range(1, 1+len(hull_masks))]
        hull_cc_overlap_stats = hull_cc_overlap_stats.sort_values('hull_size')
        hull_seg = np.zeros_like(remapped_body_seg)
        for row in hull_cc_overlap_stats.itertuples():
            mask_box, mask = hull_masks[row.hull_cc]
            view = hull_seg[box_to_slicing(*mask_box)]
            view[:] = np.where(mask, row.hull_body, view)
        update_seg_layer(viewer, 'final-hull-seg', hull_seg, scale, box)

    return mito_body_ct


def update_seg_layer(viewer, *args, **kwargs):
    if viewer is None:
        return

    from neuclease.misc.ngviewer import update_seg_layer as _update_seg_layer
    _update_seg_layer(viewer, *args, **kwargs)


def update_img_layer(viewer, *args, **kwargs):
    if viewer is None:
        return

    from neuclease.misc.ngviewer import update_img_layer as _update_img_layer
    _update_img_layer(viewer, *args, **kwargs)


def update_mask_layer(viewer, *args, **kwargs):
    if viewer is None:
        return

    from neuclease.misc.ngviewer import update_mask_layer as _update_mask_layer
    _update_mask_layer(viewer, *args, **kwargs)
