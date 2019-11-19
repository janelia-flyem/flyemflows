import os
import logging
import argparse
import datetime
from contextlib import contextmanager
from multiprocessing.pool import Pool, ThreadPool

import numpy as np
import pandas as pd
import requests

from neuclease import configure_default_logging
from neuclease.logging_setup import initialize_excepthook
from neuclease.util import tqdm_proxy, Timer, groupby_presorted, iter_batches
from neuclease.dvid import fetch_repo_info, fetch_mappings, post_labelindex_batch, fetch_labelindex, LabelIndex, create_labelindex, PandasLabelIndex, post_mappings
from flyemflows.bin.ingest_label_indexes import STATS_DTYPE, load_stats_h5_to_records, sort_block_stats

logger = logging.getLogger(__name__)

def main():
    configure_default_logging()
    initialize_excepthook()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--last-mutid', '-i', required=False, type=int)
    parser.add_argument('--num-threads', '-t', default=0, type=int,
                        help='How many threads to use when ingesting label indexes (does not currently apply to mappings)')
    parser.add_argument('--num-processes', '-p', default=0, type=int,
                        help='How many processes to use when ingesting label indexes (does not currently apply to mappings)')
    parser.add_argument('--batch-size', '-b', default=100_000, type=int,
                        help='Data is grouped in batches to the server. This is the batch size, as measured in ROWS of data to be processed for each batch.')
    parser.add_argument('server')
    parser.add_argument('src_uuid')
    parser.add_argument('dest_uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('supervoxel_block_stats_h5', nargs='?', # not required if only ingesting mapping
                        help=f'An HDF5 file with a single dataset "stats", with dtype: {STATS_DTYPE[1:]} (Note: No column for body_id)')
    args = parser.parse_args()

    with Timer() as timer:
        src_info = (args.server, args.src_uuid, args.labelmap_instance)
        dest_info = (args.server, args.dest_uuid, args.labelmap_instance)
        erase_from_labelindexes(src_info, dest_info, args.supervoxel_block_stats_h5, args.batch_size,
                                threads=args.num_threads, processes=args.num_processes, last_mutid=args.last_mutid)
    logger.info(f"DONE. Total time: {timer.timedelta}")


def erase_from_labelindexes(src_info, dest_info, erased_block_stats_h5, batch_size=1_000_000, *, threads=0, processes=0, last_mutid=None, mapping=None):
    assert not (threads and processes), \
        "Use threads or processes (or neither), but not both."
    if last_mutid is None:
        last_mutid = fetch_repo_info(*src_info[:2])["MutationID"]
    
    (block_sv_stats, presorted_by, _agglo_path) = load_stats_h5_to_records(erased_block_stats_h5)
    
    if presorted_by != 'body_id':
        if mapping is None:
            mapping = fetch_mappings(*src_info)
    
        assert isinstance(mapping, pd.Series)
        mapping_df = mapping.reset_index().rename(columns={'sv': 'segment_id', 'body': 'body_id'})
    
        # sorts in-place, and saves a copy to hdf5
        sort_block_stats( block_sv_stats,
                          mapping_df,
                          erased_block_stats_h5[:-3] + '-sorted-by-body.h5',
                          '<fetched-from-dvid>')

    if threads > 0:
        pool = ThreadPool(threads)
    elif processes > 0:
        pool = Pool(processes)
    else:
        @contextmanager
        def fakepool():
            yield
        pool = fakepool()
    
    processor = ErasedStatsBatchProcessor(last_mutid, src_info, dest_info)
    gen = generate_stats_batches(block_sv_stats, batch_size)
    progress_bar = tqdm_proxy(total=len(block_sv_stats), logger=logger)

    unexpected_dfs = []
    all_missing_bodies = []
    all_deleted_svs = []
    with progress_bar, pool:
        if threads == 0 and processes == 0:
            batch_iter = map(processor.process_batch, gen)
        else:
            batch_iter = pool.imap_unordered(processor.process_batch, gen)

        for next_stats_batch_total_rows, missing_bodies, unexpected_df, deleted_svs in batch_iter:
            if missing_bodies:
                path = 'labelindex-missing-bodies.csv'
                pd.Series(missing_bodies, name='body').to_csv(path, index=False, header=not os.path.exists(path), mode='a')
                all_missing_bodies.extend( missing_bodies )

            if unexpected_df is not None:
                path = 'labelindex-unexpected-counts.csv'
                unexpected_df.to_csv(path, index=False, header=not os.path.exists(path), mode='a')
                unexpected_dfs.append( unexpected_df )

            if len(deleted_svs) > 0:
                path = 'deleted-supervoxels.csv'
                pd.Series(deleted_svs, name='sv').to_csv(path, index=False, header=not os.path.exists(path), mode='a')
                all_deleted_svs.append(deleted_svs)

            progress_bar.update(next_stats_batch_total_rows)

    if all_deleted_svs:
        all_deleted_svs = np.concatenate(all_deleted_svs)
        assert all_deleted_svs.dtype == np.uint64

        # Now update the mapping to remove the deleted supervoxels.
        # We can't use 'batch_size' in post_mappings(...) because that function
        # still aims to group the mappings according to body, and we might have
        # a lot of body-0 mappings to post here. We have to batch these ourselves.
        for deleted_svs in iter_batches(all_deleted_svs, 100_000):
            new_mapping = pd.Series(0, index=deleted_svs, dtype=np.uint64, name='body')
            post_mappings(*dest_info, new_mapping, last_mutid)

#     mapper = LabelMapper(mapping.index.values, mapping.values)
#     changed_bodies = mapper.apply(deleted_svs, True) # @UnusedVariable
#     q = 'body in @changed_bodies and sv not in @all_deleted_svs'
#     new_mapping = mapping.reset_index().query(q)['body']
#     post_mappings(*dest_info, new_mapping, last_mutid, batch_size=100_000)




def generate_stats_batches( block_sv_stats, batch_rows=100_000 ):
    """
    Generator.
    For the given array of with dtype=STATS_DTYPE, sort the array by [body_id,z,y,x] (IN-PLACE),
    and then break it up into groups of rows with contiguous body_id.
    
    The groups are then yielded in batches, where the total rowcount across all subarrays in
    each batch has approximately batch_rows.
    
    Yields:
        (batch, batch_total_rowcount)
    """
    def gen():
        next_stats_batch = []
        next_stats_batch_total_rows = 0
    
        for batch in groupby_presorted(block_sv_stats, block_sv_stats['body_id'][:, None]):
            next_stats_batch.append( batch )
            next_stats_batch_total_rows += len(batch)
            if next_stats_batch_total_rows >= batch_rows:
                yield (next_stats_batch, next_stats_batch_total_rows)
                next_stats_batch = []
                next_stats_batch_total_rows = 0
    
        # last batch
        if next_stats_batch:
            yield (next_stats_batch, next_stats_batch_total_rows)
    
    return gen()



class ErasedStatsBatchProcessor:
    """
    Function object to take batches of grouped stats,
    convert them into the proper protobuf structure,
    and send them to an endpoint.
    
    Defined here as a class instead of a simple function to enable
    pickling (for multiprocessing), even when this file is run as __main__.
    """
    def __init__(self, last_mutid, src_info, dest_info):
        self.last_mutid = last_mutid

        self.user = os.environ.get("USER", "unknown")
        self.mod_time = datetime.datetime.now().isoformat()
        
        self.src_info = src_info
        self.dest_info = dest_info

    def process_batch(self, batch_and_rowcount):
        """
        Given a batch of ERASED block stats, fetches the existing LabelIndex,
        subtracts the erased stats, and posts either an updated labelindex or
        a tombstone (if the body is completely erased).
        """
        next_stats_batch, next_stats_batch_total_rows = batch_and_rowcount

        batch_indexes = []
        missing_bodies = []
        unexpected_dfs = []
        all_deleted_svs = []
        for body_group in next_stats_batch:
            body_id = body_group[0]['body_id']

            try:
                old_index = fetch_labelindex(*self.src_info, body_id, format='pandas')
            except requests.RequestException as ex:
                missing_bodies.append(body_id)
                if not str(ex.response.status_code).startswith('4'):
                    logger.warning(f"Failed to fetch LabelIndex for label: {body_id} due to error {ex.response.status_code}")
                continue

            old_df = old_index.blocks
            erased_df = pd.DataFrame(body_group).rename(columns={'segment_id': 'sv'})[['z', 'y', 'x', 'sv', 'count']]
            assert erased_df.columns.tolist() == old_df.columns.tolist()
            assert old_df.duplicated(['z', 'y', 'x', 'sv']).sum() == 0
            assert erased_df.duplicated(['z', 'y', 'x', 'sv']).sum() == 0
            
            # Find the rows that exist on the old side (or both)
            merged_df = old_df.merge(erased_df, 'outer', on=['z', 'y', 'x', 'sv'], suffixes=['_old', '_erased'], indicator='side')
            merged_df['count_old'] = merged_df['count_old'].fillna(0).astype(np.uint32)
            merged_df['count_erased'] = merged_df['count_erased'].fillna(0).astype(np.uint32)
            
            # If some supervoxel was "erased" from a particular block and the original
            # labelindex didn't mention it, that's a sign of corruption.
            # Save it for subsequent analysis
            unexpected_df = merged_df.query('count_old == 0').copy()
            if len(unexpected_df) > 0:
                unexpected_df['body'] = body_id
                unexpected_dfs.append(unexpected_df)

            merged_df = merged_df.query('count_old > 0').copy()
            merged_df['count'] = merged_df['count_old'] - merged_df['count_erased']

            new_df = merged_df[['z', 'y', 'x', 'sv', 'count']]
            new_df = new_df.query('count > 0').copy()

            deleted_svs = set(old_df['sv']) - set(new_df['sv'])
            if deleted_svs:
                deleted_svs = np.fromiter(deleted_svs, dtype=np.uint64)
                all_deleted_svs.append(deleted_svs)

            if len(new_df) == 0:
                # Nothing to keep. Make a tombstone.
                tombstone_index = LabelIndex()
                tombstone_index.label = body_id
                tombstone_index.last_mutid = self.last_mutid
                tombstone_index.last_mod_user = self.user
                tombstone_index.last_mod_time = self.mod_time
                batch_indexes.append(tombstone_index)
            else:
                pli = PandasLabelIndex(new_df, body_id, self.last_mutid, self.mod_time, self.user)
                new_labelindex = create_labelindex(pli)
                batch_indexes.append(new_labelindex)
        
        # Write entire batch to DVID
        post_labelindex_batch(*self.dest_info, batch_indexes)

        # Return missing body IDs and the set of unexpected rows
        if unexpected_dfs:
            unexpected_df = pd.concat(unexpected_dfs)
        else:
            unexpected_df = None
        
        if all_deleted_svs:
            all_deleted_svs = np.concatenate(all_deleted_svs)
            
        return next_stats_batch_total_rows, missing_bodies, unexpected_df, all_deleted_svs


if __name__ == "__main__":
    main()
