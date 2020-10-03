"""
Script to "correct" the centroids produced by the MitoStats workflow.

Given a segmentation source containing the mito CC objects and
a table of their centroids as exported by the MitoStats workflow,
reads the label from the segmentation under every centroid point,
to verify that it matches the corresponding mito id.
If it doesn't match, that implies the mito must have a non-convex
shape, so the centroid doesn't fall within the segment itself.
That's not technically wrong, but it's inconvenient for downstream
processing, so here we "correct" the centroid by simply picking an
arbitrary point that does lie within the mito segment.

Notes
-----

There are two steps to this procedure:

    1. Sample the label under every centroid point
    2. Fetch a block of segmentation for each "mismatched" mito
       we found, using the sparse blocks of the segment.

Since we need sparse blocks for step 2, we need to provide a dvid labelmap instance.
But for step 1, we just need a copy of the segmentation that can read points quickly.
If you only have one copy of the segmentation (in a dvid labelmap instance),
then provide it as the mito-sparsevol-source, and it will be used for both steps.
But if you happen to have a copy of the segmentation in zarr format,
you can provide it as the mito-sparsevol-source, in which case it will be used
for step 1 (and will be faster).

Example config
--------------
mito-sparsevol-source:
  dvid:
    server: emdata3:8900
    uuid: d31b64ac81444923a0319961736a6c31
    segmentation-name: masked-mito-cc
    supervoxels: true

  geometry:
    message-block-shape: [-1, -1, -1]

mito-point-source:
  zarr:
    path: /sharedscratch/flyem/bergs/v1.1/masked-mito-cc.zarr
    dataset: s1

  geometry:
    message-block-shape: [-1, -1, -1]
    block-width: -1
    available-scales: [0]

  adapters:
    # Upscale
    rescale-level: -1

Example Usage:
--------------
# On a big machine:
correct_centroids.py -s 1 -p 32 centroid-correction-config.yaml scale_0_stats_df.pkl

# As a cluster job:
bsub -n 32 -a sharedscratch -o correct-stats.log python correct_centroids.py -s 1 -p 32 centroid-correction-config.yaml scale_0_stats_df.pkl
"""
from flyemflows.volumes.adapters.scaled_volume_service import ScaledVolumeService
import sys
import pickle
import argparse
import logging
from functools import partial

logger = logging.getLogger(__name__)


def config_schema():
    from flyemflows.volumes import SegmentationVolumeSchema, DvidSegmentationVolumeSchema

    ConfigSchema = {
        "type": "object",
        "description": "Settings for the correct_centroids script",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "mito-sparsevol-source": {
                "oneOf": [
                    DvidSegmentationVolumeSchema,
                    {"type": "null"}
                ],
                "description": "Where to look for mito sparsevols, for correcting centroids. Must be a dvid source.",
                "default": None
            },
            "mito-point-source": {
                "oneOf": [
                    SegmentationVolumeSchema,
                    {"type": "null"}
                ],
                "description": "Where to look when sampling the label under each mito centroid,\n"
                               "to verify that it matches its supposed mito label.\n"
                               "If this is null, the sparsevol source will be used.\n",
                "default": {}
            }
        }
    }
    return ConfigSchema


def correct_centroids(config, stats_df, check_scale=0, verify=False, threads=0, processes=8):
    import numpy as np
    import pandas as pd

    from neuclease.util import compute_parallel, Timer
    from flyemflows.volumes import VolumeService

    with Timer("Pre-sorting points by block", logger):
        stats_df['bz'] = stats_df['by'] = stats_df['bx'] = np.int32(0)
        stats_df[['bz', 'by', 'bx']] = stats_df[[*'zyx']] // 64
        stats_df.sort_values(['bz', 'by', 'bx'], inplace=True)
        stats_df.drop(columns=['bz', 'by', 'bx'], inplace=True)

    if config['mito-sparsevol-source'] is not None:
        sparsevol_source = VolumeService.create_from_config(config['mito-sparsevol-source'])
        point_source = sparsevol_source
    else:
        sparsevol_source = None
        point_source = None

    if config['mito-point-source']:
        point_source = VolumeService.create_from_config(config['mito-point-source'])

    assert point_source or sparsevol_source, \
        "You must provide either a point-source or sparsevol-source."

    stats_df['centroid_label'] = sample_labels(point_source, stats_df, check_scale, threads, processes)

    mismatched_mitos = stats_df.query('centroid_label != mito_id')

    logger.info(f"Correcting {len(mismatched_mitos)} mismatched mito centroids")

    if sparsevol_source:
        _find_mito = partial(find_mito_from_sparsevol, *sparsevol_source.instance_triple)
        mitos_and_coords = compute_parallel(_find_mito, mismatched_mitos.index, ordered=False, threads=threads, processes=processes)
    else:
        _find_mito = partial(find_mito_from_seg, point_source, check_scale)
        mismatched_rows = mismatched_mitos.reset_index()[['mito_id', *'zyx']].astype(np.int64).values
        mitos_and_coords = compute_parallel(_find_mito, mismatched_rows, starmap=True, ordered=False, threads=threads, processes=processes)

    corrected_df = pd.DataFrame(mitos_and_coords, columns=['mito_id', *'zyx']).set_index('mito_id')
    stats_df.loc[corrected_df.index, [*'zyx']] = corrected_df[[*'zyx']]

    stats_df['centroid_type'] = 'exact'
    stats_df.loc[corrected_df.index, 'centroid_type'] = 'adjusted'

    # Sanity check: they should all be correct now!
    if verify:
        new_labels = sample_labels(point_source, stats_df.loc[mismatched_mitos.index], check_scale, threads, processes)
        if (new_labels != mismatched_mitos.index).any():
            logger.error("Some mitos remained mismstached!")

    return stats_df


def sample_labels(point_source, stats_df, check_scale, threads, processes):
    from neuclease.dvid import fetch_labels_batched
    from flyemflows.volumes import DvidVolumeService

    if isinstance(point_source, DvidVolumeService):
        return fetch_labels_batched(*point_source.instance_triple,
                                    stats_df[[*'zyx']] // (2**check_scale),
                                    supervoxels=point_source.supervoxels,
                                    scale=check_scale,
                                    batch_size=1000,
                                    threads=threads,
                                    processes=processes)

    import multiprocessing as mp
    import dask
    from dask.diagnostics import ProgressBar

    if threads:
        pool = mp.pool.ThreadPool(threads)
    else:
        pool = mp.pool.Pool(processes)

    dask.config.set(scheduler='processes')
    with pool, dask.config.set(pool=pool), ProgressBar():
        centroids = stats_df[[*'zyx']] // (2**check_scale)
        labels = point_source.sample_labels( centroids, scale=check_scale )

    return labels


def find_mito_from_sparsevol(server, uuid, instance, mito_id):
    from neuclease.dvid import generate_sample_coordinate
    coord = generate_sample_coordinate(server, uuid, instance, mito_id, supervoxels=True)
    return (mito_id, *coord)


def find_mito_from_seg(point_source, check_scale, mito_id, z, y, x):
    """
    Fetch a volume around the given point,
    and find the closest voxel that matches mito_id.
    Return the coordinate, or (0,0,0) if the subvolume does not contain mito_id at all.
    """
    import numpy as np

    for R in [50, 100, 200, 400]:
        p = np.array([z, y, x])

        p //= 2**check_scale
        R //= 2**check_scale

        subvol = point_source.get_subvolume([p-R, p+R+1], check_scale)
        c = np.argwhere(subvol == mito_id)
        c -= R
        if len(c) > 0:
            d = np.linalg.norm(c, axis=1)
            p2 = p + c[np.argmin(d)]
            p2 *= 2**check_scale
            return (mito_id, *p2)

    # Give up
    return (mito_id, 0, 0, 0)


def main():
    # Early exit if we're dumping the config
    # (Parse it ourselves to allow omission of otherwise required parameters.)
    if ({'--dump-config-template', '-d'} & {*sys.argv}):
        from confiddler import dump_default_config
        dump_default_config(config_schema(), sys.stdout)
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', '-p', type=int, default=0)
    parser.add_argument('--threads', '-t', type=int, default=0)
    parser.add_argument('--check-scale', '-s', type=int, default=0)
    parser.add_argument('--dump-config-template', '-d', action='store_true')
    parser.add_argument('--verify', '-v', action='store_true')
    parser.add_argument('config')
    parser.add_argument('stats_df_pkl', help='Mito statistics table, as produced by the MitoStats workflow.'
                                             'Note: The coordinates must be provided in scale-0 units,'
                                             '      regardless of the check-scale you want to use!')
    args = parser.parse_args()

    if args.threads == 0 and args.processes == 0:
        args.threads = 1
    elif (args.threads != 0) and (args.processes != 0):
        raise RuntimeError("Can't use multi-threading and multi-processing.  Pick one.")

    from neuclease import configure_default_logging
    configure_default_logging()

    from confiddler import load_config

    config = load_config(args.config, config_schema())

    with open(args.stats_df_pkl, 'rb') as f:
        stats_df = pickle.load(f)

    stats_df = correct_centroids(config,
                                 stats_df,
                                 check_scale=args.check_scale,
                                 verify=args.verify,
                                 threads=args.threads,
                                 processes=args.processes)

    with open('corrected_stats_df.pkl', 'wb') as f:
        pickle.dump(stats_df, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
