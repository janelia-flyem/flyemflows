import os
import logging
import datetime
from contextlib import contextmanager
from multiprocessing.pool import Pool, ThreadPool

import numpy as np
import pandas as pd
import requests

from dvidutils import LabelMapper
from neuclease.util import tqdm_proxy
from neuclease.dvid import fetch_repo_info, fetch_mappings, post_labelindex_batch, fetch_labelindex, LabelIndex, create_labelindex, PandasLabelIndex, post_mappings
from flyemflows.bin.ingest_label_indexes import load_stats_h5_to_records, generate_stats_batches, sort_block_stats

logger = logging.getLogger(__name__)

def main():
    assert False, "FIXME"


def erase_from_labelindexes(src_info, dest_info, erased_block_stats_h5, batch_size=1_000_000, *, threads=0, processes=0, last_mutid=None, mapping=None):
    assert not processes, "Multiprocessing in this function isn't stable yet. Use threads."
    assert not (threads and processes), \
        "You can use threads or processes (or neither), but not both"

    if last_mutid is None:
        last_mutid = fetch_repo_info(*src_info[:2])["MutationID"]
    
    (block_sv_stats, presorted_by, _agglo_path) = load_stats_h5_to_records(erased_block_stats_h5)
    
    if mapping is None:
        mapping = fetch_mappings(*src_info)

    assert isinstance(mapping, pd.Series)
    mapping_df = mapping.reset_index().rename(columns={'sv': 'segment_id', 'body': 'body_id'})

    if presorted_by != 'body_id':
        # sorts in-place, and saves a copy to hdf5
        sort_block_stats( block_sv_stats,
                          mapping_df,
                          erased_block_stats_h5[:-3] + '-sorted-by-body.h5',
                          '<fetched-from-dvid>')

    # 'processor' is declared as a global so it can be shared with
    # subprocesses quickly via implicit memory sharing via after fork()
    global processor
    processor = ErasedStatsBatchProcessor(last_mutid, src_info, dest_info, block_sv_stats)
    gen = generate_stats_batches(block_sv_stats, batch_size)
    progress_bar = tqdm_proxy(total=len(block_sv_stats), logger=logger)

    # Pool must be created AFTER processor is instantiated, above,
    # to inherit it via fork()
    if threads > 0:
        pool = ThreadPool(threads)
    elif processes > 0:
        pool = Pool(processes)
    else:
        @contextmanager
        def fakepool():
            yield
        pool = fakepool
    
    unexpected_dfs = []
    all_missing_bodies = []
    all_deleted_svs = []
    with progress_bar, pool:
        if threads == 0 and processes == 0:
            batch_iter = map(processor.process_batch, gen)
        else:
            # Rather than call pool.imap_unordered() with processor.process_batch(),
            # we use globally declared process_batch(), as explained below.
            batch_iter = pool.imap_unordered(process_batch, gen)

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

    # Now update the mapping to remove the deleted supervoxels.
    mapper = LabelMapper(mapping.index.values, mapping.values)

    if all_deleted_svs:
        all_deleted_svs = np.concatenate(all_deleted_svs)
        assert all_deleted_svs.dtype == np.uint64

    changed_bodies = mapper.apply(deleted_svs, True) # @UnusedVariable
    q = 'body in @changed_bodies and sv not in @all_deleted_svs'
    new_mapping = mapping.reset_index().query(q)['body']
    post_mappings(*dest_info, new_mapping, last_mutid, batch_size=100_000)


# This is a dirty little trick:
# We declare 'processor' and 'process_batch()' as a globals to avoid
# having it get implicitly pickled it when passing to subprocesses.
# The memory for block_sv_stats is thus inherited by
# child processes implicitly, via fork().
processor = None
def process_batch(*args):
    return processor.process_batch(*args)


class ErasedStatsBatchProcessor:
    """
    Function object to take batches of grouped stats,
    convert them into the proper protobuf structure,
    and send them to an endpoint.
    
    Defined here as a class instead of a simple function to enable
    pickling (for multiprocessing), even when this file is run as __main__.
    """
    def __init__(self, last_mutid, src_info, dest_info, block_sv_stats):
        self.last_mutid = last_mutid
        self.block_sv_stats = block_sv_stats

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
        for body_group_start, body_group_stop in next_stats_batch:
            body_group = self.block_sv_stats[body_group_start:body_group_stop]
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
