import os
import logging
import argparse

import h5py
import numpy as np
import pandas as pd

from neuclease import configure_default_logging
from neuclease.util import groupby_spans_presorted, iter_batches, tqdm_proxy, Timer
from neuclease.logging_setup import initialize_excepthook
from neuclease.dvid import fetch_mappings
from flyemflows.bin.ingest_label_indexes import STATS_DTYPE, load_stats_h5_to_records, sort_block_stats

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-into-batches', type=int,
                        help='If given, also split the body stats into this many batches of roughly equal size')
    parser.add_argument('server')
    parser.add_argument('src_uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('supervoxel_block_stats_h5',
                        help=f'An HDF5 file with a single dataset "stats", with dtype: {STATS_DTYPE[1:]} (Note: No column for body_id)')
    args = parser.parse_args()

    configure_default_logging()
    initialize_excepthook()
    (block_sv_stats, _presorted_by, _agglo_path) = load_stats_h5_to_records(args.supervoxel_block_stats_h5)

    src_info = (args.server, args.src_uuid, args.labelmap_instance)
    mapping = fetch_mappings(*src_info)

    assert isinstance(mapping, pd.Series)
    mapping_df = mapping.reset_index().rename(columns={'sv': 'segment_id', 'body': 'body_id'})

    # sorts in-place, and saves a copy to hdf5
    sort_block_stats( block_sv_stats,
                      mapping_df,
                      args.supervoxel_block_stats_h5[:-3] + '-sorted-by-body.h5',
                      '<fetched-from-dvid>')
    
    if args.split_into_batches:
        num_batches = args.split_into_batches
        batch_size = int(np.ceil(len(block_sv_stats) / args.split_into_batches))
        logger.info(f"Splitting into {args.split_into_batches} batches of size ~{batch_size}")
        os.makedirs('stats-batches', exist_ok=True)
        
        body_spans = groupby_spans_presorted(block_sv_stats['body_id'][:, None])
        for batch_index, batch_spans in enumerate(tqdm_proxy(iter_batches(body_spans, batch_size))):
            span_start, span_stop = batch_spans[0][0], batch_spans[-1][1]
            batch_stats = block_sv_stats[span_start:span_stop]
            digits = int(np.ceil(np.log10(num_batches)))
            batch_path = f'stats-batches/stats-batch-{batch_index:0{digits}d}.h5'
            save_stats(batch_stats, batch_path)
    
    logger.info("DONE sorting stats by body")


def save_stats(block_sv_stats, output_path):
    with Timer(f"Saving sorted stats to {output_path}"), h5py.File(output_path, 'w') as f:
        f.create_dataset('stats', data=block_sv_stats, chunks=True)
        f['stats'].attrs['presorted-by'] = 'body_id'
        f['stats'].attrs['agglomeration-mapping-path'] = '<unset>'


if __name__ == "__main__":
    main()
