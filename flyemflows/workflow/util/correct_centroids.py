import sys
import pickle
import argparse
import logging
from functools import partial

logger = logging.getLogger(__name__)

def config_schema():
    from flyemflows.volumes import VolumeService, SegmentationVolumeSchema, DvidSegmentationVolumeSchema, DvidVolumeService, ScaledVolumeService

    ConfigSchema = {
        "type": "object",
        "description": "Settings for the correct_centroids script",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "mito-sparsevol-source": {
                **DvidSegmentationVolumeSchema,
                "description": "Where to look for mito sparsevols, for correcting centroids. Must be a dvid source."
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

    from neuclease.util import tqdm_proxy, compute_parallel, Timer
    from neuclease.dvid import fetch_labels_batched
    from flyemflows.volumes import VolumeService, DvidVolumeService

    with Timer("Pre-sorting points by block", logger):
        stats_df['bz'] = stats_df['by'] = stats_df['bx'] = np.int32(0)
        stats_df[['bz', 'by', 'bx']] = stats_df[[*'zyx']] // 64
        stats_df.sort_values(['bz', 'by', 'bx'], inplace=True)
        stats_df.drop(columns=['bz', 'by', 'bx'], inplace=True)

    sparsevol_source = VolumeService.create_from_config(config['mito-sparsevol-source'])
    if config['mito-point-source'] is None:
        point_source = sparsevol_source
    else:
        point_source = VolumeService.create_from_config(config['mito-point-source'])

    if isinstance(point_source, DvidVolumeService):
        stats_df['centroid_label'] = fetch_labels_batched(*point_source.instance_triple,
                                                          stats_df[[*'zyx']] // (2**check_scale),
                                                          supervoxels=point_source.supervoxels,
                                                          scale=check_scale,
                                                          batch_size=1000,
                                                          threads=threads,
                                                          processes=processes)
    else:
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
            stats_df['centroid_label'] = point_source.sample_labels( centroids, scale=check_scale )

    mismatched_mitos = stats_df.query('centroid_label != mito_id').index

    logger.info(f"Correcting {len(mismatched_mitos)} mismatched mito centroids")
    _find_mito = partial(find_mito, *sparsevol_source.instance_triple)
    mitos_and_coords = compute_parallel(_find_mito, mismatched_mitos, ordered=False, threads=threads, processes=processes)
    corrected_df = pd.DataFrame(mitos_and_coords, columns=['mito_id', *'zyx']).set_index('mito_id')
    stats_df.loc[corrected_df.index, [*'zyx']] = corrected_df[[*'zyx']]
    stats_df.loc[corrected_df.index, 'centroid_type'] = 'adjusted'

    # Sanity check: they should all be correct now!
    if verify:
        new_centroids = stats_df.loc[mismatched_mitos, [*'zyx']].values
        new_labels = fetch_labels_batched(*sparsevol_source.instance_triple,
                                          new_centroids,
                                          supervoxels=True,
                                          threads=threads,
                                          processes=processes)

        if (new_labels != mismatched_mitos).any():
            logger.error("Some mitos remained mismstached!")

    return stats_df


def find_mito(server, uuid, instance, mito_id):
    from neuclease.dvid import generate_sample_coordinate
    coord = generate_sample_coordinate(server, uuid, instance, mito_id, supervoxels=True)
    return (mito_id, *coord)


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
    parser.add_argument('stats_df_pkl')
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
