#!/usr/bin/env python3
import os
import sys
import argparse
import datetime
import logging
import multiprocessing
from itertools import chain
    
import requests

import h5py
import numpy as np
import pandas as pd
from numba import jit

from neuclease.dvid import fetch_repo_info, fetch_instance_info, fetch_repo_instances, post_labelindex_batch, fetch_labelindex, fetch_complete_mappings, post_mappings
from neuclease.dvid.labelmap.labelops_pb2 import LabelIndex

from dvidutils import LabelMapper # Fast label mapping in C++

from neuclease import configure_default_logging
from neuclease.util import Timer, groupby_presorted, groupby_spans_presorted, tqdm_proxy
from neuclease.logging_setup import initialize_excepthook
from neuclease.merge_table import load_edge_csv

from flyemflows.workflow.util.config_helpers import load_body_list

logger = logging.getLogger(__name__)


STATS_DTYPE = [('body_id',    np.uint64),
               ('segment_id', np.uint64),
               ('z',          np.int32),
               ('y',          np.int32),
               ('x',          np.int32),
               ('count',      np.uint32)]

AGGLO_MAP_COLUMNS = ['segment_id', 'body_id']


def main():
    """
    Command-line wrapper interface for ingest_label_indexes(), and/or ingest_mapping(), below.
    """
    configure_default_logging()
    initialize_excepthook()
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--last-mutid', '-i', required=False, type=int)
    parser.add_argument('--check-mismatches', action='store_true',
                        help='If given, every LabelIndex will be compared with the existing LabelIndex on the server, and only the mismatching ones will be sent.')
    parser.add_argument('--agglomeration-mapping', '-m', required=False,
                        help='Either a UUID to pull the mapping from, or a CSV file with two columns, mapping supervoxels to agglomerated bodies. Any missing entries implicitly identity-mapped.')
    parser.add_argument('--operation', default='indexes', choices=['indexes', 'mappings', 'both', 'sort-only'],
                        help='Whether to load the LabelIndices, MappingOps, or both. If sort-only, sort/save the stats and exit.')
    parser.add_argument('--subset-labels', required=False,
                        help='CSV file with a single column of label IDs to write LabelIndexes for.'
                             'Other labels found in the mapping and or block stats h5 file will be ignored. '
                             'NOTE: Whether or not the label ids are interpreted as supervoxels or bodies depends on whether or not --agglomeration-mapping was provided.')
    parser.add_argument('--tombstones', default='include', choices=['include', 'exclude', 'only'],
                        help="Whether to include 'tombstones' in the labelindexes (i.e. explicitly send empty labelindexes for all supervoxels in a body that don't match the body-id). "
                             "Options are 'include', 'exclude', or 'only' (i.e. send only the tombstones and not the actual labelindices)")
    parser.add_argument('--num-threads', '-n', default=1, type=int,
                        help='How many threads to use when ingesting label indexes (does not currently apply to mappings)')
    parser.add_argument('--batch-size', '-b', default=20_000, type=int,
                        help='Data is grouped in batches to the server. This is the batch size, as measured in ROWS of data to be processed for each batch.')
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('supervoxel_block_stats_h5', nargs='?', # not required if only ingesting mapping
                        help=f'An HDF5 file with a single dataset "stats", with dtype: {STATS_DTYPE[1:]} (Note: No column for body_id)')

    args = parser.parse_args()

    with Timer() as timer:
        main_impl(args)
    logger.info(f"DONE. Total time: {timer.timedelta}")


def main_impl(args):
    # Read agglomeration file
    segment_to_body_df = None
    if args.agglomeration_mapping:
        with Timer("Loading agglomeration mapping", logger):
            if args.agglomeration_mapping.endswith('.csv'):
                mapping_pairs = load_edge_csv(args.agglomeration_mapping)
                segment_to_body_df = pd.DataFrame(mapping_pairs, columns=AGGLO_MAP_COLUMNS)
            else:
                if set(args.agglomeration_mapping) - set('0123456789abcdef'):
                    raise RuntimeError(f"Your agglomeration mapping is neither a CSV file nor a UUID: {args.agglomeration_mapping}")

                mapping_uuid = args.agglomeration_mapping
                logger.info(f"Loading agglomeration mapping from UUID {mapping_uuid}")
                mapping_series = fetch_complete_mappings(args.server, mapping_uuid, args.labelmap_instance)
                segment_to_body_df = pd.DataFrame( {'segment_id': mapping_series.index.values} )
                segment_to_body_df['body_id'] = mapping_series.values
                assert (segment_to_body_df.columns == AGGLO_MAP_COLUMNS).all()

    subset_labels = None
    if args.subset_labels:
        is_supervoxels = (args.agglomeration_mapping is None)
        subset_labels = load_body_list(args.subset_labels, is_supervoxels)
        subset_labels = set(subset_labels)

    if args.last_mutid is None:
        args.last_mutid = fetch_repo_info(args.server, args.uuid)['MutationID']

    # Upload label indexes
    if args.operation in ('indexes', 'both', 'sort-only'):
        if not args.supervoxel_block_stats_h5:
            raise RuntimeError("You must provide a supervoxel_block_stats_h5 file if you want to ingest LabelIndexes")

        # Read block stats file
        block_sv_stats, presorted_by, agglomeration_path = load_stats_h5_to_records(args.supervoxel_block_stats_h5)
        
        stats_are_presorted = False
        if args.agglomeration_mapping:
            if (presorted_by == 'body_id') and (agglomeration_path == args.agglomeration_mapping):
                stats_are_presorted = True
        elif presorted_by == 'segment_id':
            stats_are_presorted = True
        
        if stats_are_presorted:
            logger.info("Stats are pre-sorted")
        else:
            output_dir, basename = os.path.split(os.path.abspath(args.supervoxel_block_stats_h5))
            if segment_to_body_df is None:
                output_path = output_dir + '/sorted-by-segment-' + basename
            else:
                output_path = output_dir + '/sorted-by-body-' +  basename
            sort_block_stats(block_sv_stats, segment_to_body_df, output_path, args.agglomeration_mapping)
    
        if args.operation == 'sort-only':
            return

        with Timer(f"Grouping {len(block_sv_stats)} blockwise supervoxel counts and loading LabelIndices", logger):
            ingest_label_indexes( args.server,
                                  args.uuid,
                                  args.labelmap_instance,
                                  args.last_mutid,
                                  block_sv_stats,
                                  subset_labels,
                                  args.tombstones,
                                  batch_rows=args.batch_size,
                                  num_threads=args.num_threads,
                                  check_mismatches=args.check_mismatches )

    # Upload mappings
    if args.operation in ('mappings', 'both'):
        if not args.agglomeration_mapping:
            raise RuntimeError("Can't load mappings without an agglomeration-mapping file.")
        
        with Timer(f"Loading mapping ops", logger):
            ingest_mapping( args.server,
                            args.uuid,
                            args.labelmap_instance,
                            args.last_mutid,
                            segment_to_body_df,
                            subset_labels,
                            args.batch_size )

def sort_block_stats(block_sv_stats, segment_to_body_df=None, output_path=None, agglo_mapping_path=None):
    """
    Sorts the block stats by body ID, IN-PLACE.
    If segment_to_body_df is given, the body_id column is overwritten with mapped IDs.
    If agglo_mapping_path and output_path are given, save the sorted result to an hdf5 file.

    block_sv_stats:
        numpy structured array of blockwise supervoxel counts, with dtype:
        ['body_id', 'segment_id', 'z', 'y', 'x', 'count'].

    segment_to_body_df:
        If loading an agglomeration, must be a 2-column DataFrame, mapping supervoxel-to-body.
        If loading unagglomerated supervoxels, set to None (identity mapping is used).

    output_path:
        If given, sorted result will be saved as hdf5 to this file,
        with the internal dataset name 'stats'

    agglo_mapping_path:
        A path indicating where the segment_to_body_df was loaded from.
        It's saved to the hdf5 attributes for provenance tracking.
    
    """
    with Timer("Assigning body IDs", logger):
        _overwrite_body_id_column(block_sv_stats, segment_to_body_df)

    with Timer(f"Sorting {len(block_sv_stats)} block stats", logger):
        block_sv_stats.sort(order=['body_id', 'z', 'y', 'x', 'segment_id', 'count'])

    if output_path:
        with Timer(f"Saving sorted stats to {output_path}"), h5py.File(output_path, 'w') as f:
            f.create_dataset('stats', data=block_sv_stats, chunks=True)
            if segment_to_body_df is None:
                f['stats'].attrs['presorted-by'] = 'segment_id'
            else:
                assert agglo_mapping_path
                f['stats'].attrs['presorted-by'] = 'body_id'
                f['stats'].attrs['agglomeration-mapping-path'] = agglo_mapping_path
                

def ingest_mapping( server,
                    uuid,
                    instance,
                    mutid,
                    segment_to_body_df,
                    subset_labels=None,
                    batch_size=100_000 ):
    """
    Ingest the forward-map (supervoxel-to-body) into DVID via the .../mappings endpoint
    
    Args:
        server, uuid, instance_name:
            DVID instance info
    
        mutid:
            The mutation ID to use for all mappings
        
        segment_to_body_df:
            DataFrame.  Must have columns ['segment_id', 'body_id']
        
        batch_size:
            Approximately how many mapping pairs to pack into a single REST call.
    
    """
    assert list(segment_to_body_df.columns) == AGGLO_MAP_COLUMNS
    if fetch_repo_instances(server, uuid)[instance] != 'labelmap':
        raise RuntimeError(f"DVID instance is not a labelmap: {instance}")

    segment_to_body_df.sort_values(['body_id', 'segment_id'], inplace=True)
    
    if subset_labels is not None:
        segment_to_body_df = segment_to_body_df.query('body_id in @subset_labels')
        if len(segment_to_body_df) == 0:
            raise RuntimeError("None of the selected bodies have any mappings to post!")
    
    mappings = segment_to_body_df.set_index('segment_id')['body_id']
    post_mappings(server, uuid, instance, mappings, mutid, batch_size=batch_size)


def ingest_label_indexes( server,
                          uuid,
                          instance_name,
                          last_mutid,
                          sorted_block_sv_stats,
                          subset_labels=None,
                          tombstone_mode='include',
                          batch_rows=1_000_000,
                          num_threads=1,
                          check_mismatches=False ):
    """
    Ingest the label indexes for a particular agglomeration.
    
    Args:
        server, uuid, instance_name:
            DVID instance info
    
        last_mutid:
            The mutation ID to use for all indexes
        
        sorted_block_sv_stats:
            numpy structured array of PRE-SORTED blockwise supervoxel counts, with dtype:
            ['body_id', 'segment_id', 'z', 'y', 'x', 'count'].
        
        tombstone_mode:
            Whether or not to include tombstones in the result, or possibly ONLY tombstones, and not the 'real' labelindices.
            Choices: 'include', 'exclude', or 'only'.
        
        batch_size:
            How many LabelIndex structures to include in each /indices REST call.
        
        num_threads:
            How many threads to use, for parallel loading.
    """
    _check_instance(server, uuid, instance_name)
    block_sv_stats = sorted_block_sv_stats

    # 'processor' is declared as a global so it can be shared with
    # subprocesses quickly via implicit memory sharing via after fork()
    global processor
    instance_info = (server, uuid, instance_name)
    processor = StatsBatchProcessor(last_mutid, instance_info, tombstone_mode, block_sv_stats, subset_labels, check_mismatches)

    gen = generate_stats_batches(block_sv_stats, batch_rows)

    progress_bar = tqdm_proxy(total=len(block_sv_stats), logger=logger)

    all_mismatch_ids = []
    all_missing_ids = []
    pool = multiprocessing.Pool(num_threads)
    with progress_bar, pool:
        # Rather than call pool.imap_unordered() with processor.process_batch(),
        # we use globally declared process_batch(), as explained below.
        for next_stats_batch_total_rows, batch_mismatches, batch_missing in pool.imap_unordered(process_batch, gen):
            if batch_mismatches:
                pd.Series(batch_mismatches).to_csv(f'labelindex-mismatches-{uuid}.csv', index=False, header=False, mode='a')
                all_mismatch_ids.extend( batch_mismatches )

            if batch_missing:
                pd.Series(batch_missing).to_csv(f'labelindex-missing-{uuid}.csv', index=False, header=False, mode='a')
                all_missing_ids.extend( batch_missing )
            progress_bar.update(next_stats_batch_total_rows)

    if check_mismatches:
        logger.info(f"Mismatched LabelIndex count: {len(all_mismatch_ids)}")
        logger.info(f"Missing LabelIndex count: {len(all_missing_ids)}")


def _check_instance(server, uuid, instance):
    """
    Verify that the instance is a valid destination for the LabelIndices we're about to ingest.
    """
    if fetch_repo_instances(server, uuid)[instance] != 'labelmap':
        raise RuntimeError(f"DVID instance is not a labelmap: {instance}")

    info = fetch_instance_info(server, uuid, instance)
    bz, by, bx = info["Extended"]["BlockSize"]
    assert bz == by == bx == 64, \
        "The code below makes the hard-coded assumption that the instance block width is 64."

def generate_stats_batches( block_sv_stats, batch_rows=100_000 ):
    """
    Generator.
    For the given array of with dtype=STATS_DTYPE, sort the array by [body_id,z,y,x] (IN-PLACE),
    and then break it up into spans of rows with contiguous body_id.
    
    The spans are then yielded in batches, where the total rowcount across all subarrays in
    each batch has approximately batch_rows.
    
    Yields:
        (batch, batch_total_rowcount)
    """
    def gen():
        next_stats_batch = []
        next_stats_batch_total_rows = 0
    
        # Here, 'span' is a tuple: (start_row, stop_row)
        for span in groupby_spans_presorted(block_sv_stats['body_id'][:, None]):
            next_stats_batch.append( span )
            next_stats_batch_total_rows += (span[1] - span[0])
            if next_stats_batch_total_rows >= batch_rows:
                yield (next_stats_batch, next_stats_batch_total_rows)
                next_stats_batch = []
                next_stats_batch_total_rows = 0
    
        # last batch
        if next_stats_batch:
            yield (next_stats_batch, next_stats_batch_total_rows)

    return gen()


def _overwrite_body_id_column(block_sv_stats, segment_to_body_df=None):
    """
    Given a stats array with 'columns' as defined in STATS_DTYPE,
    overwrite the body_id column according to the given agglomeration mapping DataFrame.
    
    If no mapping is given, simply copy the segment_id column into the body_id column.
    
    Args:
        block_sv_stats: numpy.ndarray, with dtype=STATS_DTYPE
        segment_to_body_df: pandas.DataFrame, with columns ['segment_id', 'body_id']
    """
    assert block_sv_stats.dtype == STATS_DTYPE

    assert STATS_DTYPE[0][0] == 'body_id'
    assert STATS_DTYPE[1][0] == 'segment_id'
    
    block_sv_stats = block_sv_stats.view( [STATS_DTYPE[0], STATS_DTYPE[1], ('other_cols', STATS_DTYPE[2:])] )

    if segment_to_body_df is None:
        # No agglomeration
        block_sv_stats['body_id'] = block_sv_stats['segment_id']
    else:
        assert list(segment_to_body_df.columns) == AGGLO_MAP_COLUMNS
        
        # This could be done via pandas merge(), followed by fillna(), etc.,
        # but I suspect LabelMapper is faster and more frugal with RAM.
        mapper = LabelMapper(segment_to_body_df['segment_id'].values, segment_to_body_df['body_id'].values)
        del segment_to_body_df
    
        # Remap in batches to save RAM
        batch_size = 1_000_000
        for chunk_start in range(0, len(block_sv_stats), batch_size):
            chunk_stop = min(chunk_start+batch_size, len(block_sv_stats))
            chunk_segments = block_sv_stats['segment_id'][chunk_start:chunk_stop]
            block_sv_stats['body_id'][chunk_start:chunk_stop] = mapper.apply(chunk_segments, allow_unmapped=True)


# This is a dirty little trick:
# We declare 'processor' and 'process_batch()' as a globals to avoid
# having it get implicitly pickled it when passing to subprocesses.
# The memory for block_sv_stats is thus inherited by
# child processes implicitly, via fork().
processor = None
def process_batch(*args):
    return processor.process_batch(*args)

class StatsBatchProcessor:
    """
    Function object to take batches of grouped stats,
    convert them into the proper protobuf structure,
    and send them to an endpoint.
    
    Defined here as a class instead of a simple function to enable
    pickling (for multiprocessing), even when this file is run as __main__.
    """
    def __init__(self, last_mutid, instance_info, tombstone_mode, block_sv_stats, subset_labels=None, check_mismatches=False):
        assert tombstone_mode in ('include', 'exclude', 'only')
        self.last_mutid = last_mutid
        self.tombstone_mode = tombstone_mode
        self.block_sv_stats = block_sv_stats
        self.check_mismatches = check_mismatches
        self.subset_labels = subset_labels

        self.user = os.environ.get("USER", "unknown")
        self.mod_time = datetime.datetime.now().isoformat()
        
        server, uuid, instance = instance_info
        if not server.startswith('http://'):
            server = 'http://' + server
            instance_info = (server, uuid, instance)
        self.instance_info = instance_info


    def process_batch(self, batch_and_rowcount):
        """
        Takes a batch of grouped stats rows and sends it to dvid in the appropriate protobuf format.
        
        If self.check_mismatches is True, read the labelindex for each 
        """
        next_stats_batch, next_stats_batch_total_rows = batch_and_rowcount
        labelindex_batch = chain(*map(self.label_indexes_for_body, next_stats_batch))

        if not self.check_mismatches:
            post_labelindex_batch(*self.instance_info, labelindex_batch)
            return next_stats_batch_total_rows, [], []

        # Check for mismatches
        mismatch_batch = []
        missing_batch = []
        for labelindex in labelindex_batch:
            try:
                existing_labelindex = fetch_labelindex(*self.instance_info, labelindex.label)
            except requests.RequestException as ex:
                missing_batch.append(labelindex)
                if not str(ex.response.status_code).startswith('4'):
                    logger.warning(f"Failed to fetch LabelIndex for label: {labelindex.label} due to error {ex.response.status_code}")
            else:
                if (labelindex.blocks != existing_labelindex.blocks):
                    # Update the mut_id to match the previous one.
                    labelindex.last_mutid = existing_labelindex.last_mutid
                    mismatch_batch.append(labelindex)

        # Post mismatches (only)
        post_labelindex_batch(*self.instance_info, mismatch_batch + missing_batch)

        # Return mismatch IDs
        mismatch_labels = [labelindex.label for labelindex in mismatch_batch]
        missing_labels = [labelindex.label for labelindex in missing_batch]
        
        return next_stats_batch_total_rows, mismatch_labels, missing_labels


    def label_indexes_for_body(self, body_group_span):
        """
        Load body_group (a subarray with dtype=STATS_DTYPE
        and a only single unique body_id) into a LabelIndex protobuf structure.
        """
        label_indexes = []
        
        body_group_start, body_group_stop = body_group_span
        body_group = self.block_sv_stats[body_group_start:body_group_stop]
        body_id = body_group[0]['body_id']
        
        if (self.subset_labels is not None) and (body_id not in self.subset_labels):
            return []

        if self.tombstone_mode != 'only':
            label_index = LabelIndex()
            label_index.label = body_id
            label_index.last_mutid = self.last_mutid
            label_index.last_mod_user = self.user
            label_index.last_mod_time = self.mod_time
            
            body_dtype = STATS_DTYPE[0]
            segment_dtype = STATS_DTYPE[1]
            coords_dtype = ('coord_cols', STATS_DTYPE[2:5])
            count_dtype = STATS_DTYPE[5]
            assert body_dtype[0] == 'body_id'
            assert segment_dtype[0] == 'segment_id'
            assert np.dtype(coords_dtype[1]).names == ('z', 'y', 'x')
            assert count_dtype[0] == 'count'
            
            body_group = body_group.view([body_dtype, segment_dtype, coords_dtype, count_dtype])
            coord_cols = body_group['coord_cols'].view((np.int32, 3)).reshape(-1, 3)
            for block_group in groupby_presorted(body_group, coord_cols):
                coord = block_group['coord_cols'][0]
                encoded_block_id = _encode_block_id(coord)
                label_index.blocks[encoded_block_id].counts.update( zip(block_group['segment_id'], block_group['count']) )
    
            label_indexes.append(label_index)
        
        if self.tombstone_mode in ('include', 'only'):
            # All segments in this body should no longer get a real index
            # (except for the segment that matches the body_id itself).
            # We'll send an empty LabelIndex (a 'tombstone') for each one.
            all_segments = np.unique(body_group['segment_id'])
            tombstone_segments = all_segments[all_segments != body_id]
            for segment_id in tombstone_segments:
                tombstone_index = LabelIndex()
                tombstone_index.label = segment_id
                tombstone_index.last_mutid = self.last_mutid
                tombstone_index.last_mod_user = self.user
                tombstone_index.last_mod_time = self.mod_time
                label_indexes.append(tombstone_index)

        return label_indexes


@jit(nopython=True)
def _encode_block_id(coord):
    """
    Helper function for StatsBatchProcessor.label_index_for_body().
    Encodes a coord (structured array of z,y,x)
    into a uint64, in the format DVID expects.
    """
    encoded_block_id = np.uint64(0)
    encoded_block_id |= np.uint64(coord.z // 64) << 42
    encoded_block_id |= np.uint64(coord.y // 64) << 21
    encoded_block_id |= np.uint64(coord.x // 64)
    return encoded_block_id


def load_stats_h5_to_records(h5_path):
    """
    Read a block segment statistics HDF5 file.
    The file should contain a dataset named 'stats', whose dtype
    is the same as STATS_DTYPE, but possibly without a 'body_id' column. 

    If the dataset contains no 'body_id' column,
    one is prepended to the result (as a copy of the segment_id column).
    
    Returns:
        (block_sv_stats, presorted_by, agglomeration_path)
        
        where:
            block_sv_stats:
                ndarray with dtype=STATS_DTYPE
            
            presorted_by:
                One of the following:
                    - None: stats are not sorted
                    - 'segment_id': stats were sorted by the 'segment_id' column
                    - 'body_id': stats were sorted by the 'body_id' column

            agglomeration_path:
                A path pointing to the agglomeration mapping which was used to produce the 'body_id' column when the file was saved.
    """
    with h5py.File(h5_path, 'r') as f:
        dset = f['stats']
        with Timer(f"Allocating RAM for {len(dset)} block stats rows", logger):
            block_sv_stats = np.empty(dset.shape, dtype=STATS_DTYPE)

        if 'body_id' in dset.dtype.names:
            dest_view = block_sv_stats
        else:
            full_view = block_sv_stats.view([('body_col', [STATS_DTYPE[0]]), ('other_cols', STATS_DTYPE[1:])])
            dest_view = full_view['other_cols']

        with Timer(f"Loading block stats into RAM", logger):
            h5_batch_size = 1_000_000
            for batch_start in range(0, len(dset), h5_batch_size):
                batch_stop = min(batch_start + h5_batch_size, len(dset))
                dest_view[batch_start:batch_stop] = dset[batch_start:batch_stop]

        if 'body_id' not in dset.dtype.names:
            block_sv_stats['body_id'] = block_sv_stats['segment_id']
    
        try:
            presorted_by = dset.attrs['presorted-by']
            assert presorted_by in ('segment_id', 'body_id')
        except KeyError:
            presorted_by = None
    
        agglomeration_path = None
        if presorted_by == 'body_id':
            agglomeration_path = dset.attrs['agglomeration-mapping-path']
        
    return block_sv_stats, presorted_by, agglomeration_path


def group_sums_presorted(a, sorted_cols):
    """
    Args:
        a: Columns to aggregate

        sorted_cols: Columns to group by

        agg_func: Aggregation function.
                  Must accept array as the first arg, and 'axis' as a keyword arg. 
    """
    assert a.ndim >= 2
    assert sorted_cols.ndim >= 2
    assert a.shape[0] == sorted_cols.shape[0]

    # Two passes: first to get len
    @jit(nopython=True)
    def count_groups():
        num_groups = 0
        for _ in groupby_presorted(a, sorted_cols):
            num_groups += 1
        return num_groups

    num_groups = count_groups()
    print(f"Aggregating {num_groups} groups")
    
    groups_shape = (num_groups,) + sorted_cols.shape[1:]
    groups = np.zeros(groups_shape, dtype=sorted_cols.dtype)

    results_shape = (num_groups,) + a.shape[1:]
    agg_results = np.zeros(results_shape, dtype=a.dtype)
    
    @jit(nopython=True)
    def _agg(a, sorted_cols, groups, agg_results):
        pos = 0
        for i, group_rows in enumerate(groupby_presorted(a, sorted_cols)):
            groups[i] = sorted_cols[pos]
            pos += len(group_rows)
            agg_results[i] = group_rows.sum(0) # axis 0
        return (groups, agg_results)

    return _agg(a, sorted_cols, groups, agg_results)


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        print("Starting with debug arguments")
        os.chdir('/tmp/compute-frankenbody-blockstats-20190227.013919')
        sys.argv += """
              --operation=mappings \
              --agglomeration-mapping=frankenbody-mapping-after-cc.csv \
              --subset-labels=frankenbody-label.csv \
              --tombstones=include \
              --num-threads=4 \
              emdata1:8900 \
              c6591f6368eb4e96b255e11087781e6b \
              segmentation \
              block-statistics.h5 \
            """.split()
        
    if False:
        import DVIDSparkServices
        os.chdir(os.path.dirname(DVIDSparkServices.__file__) + '/..')
        
        import yaml
        test_dir = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/test_copyseg/temp_data'
        with open(f'{test_dir}/config.yaml', 'r') as f:
            config = yaml.load(f)

        dvid_config = config['outputs'][0]['dvid']
        
        ##
        mapping_file = f'{test_dir}/../LABEL-TO-BODY-mod-100-labelmap.csv'
        block_stats_file = f'{test_dir}/block-statistics.h5'

        # SPECIAL DEBUGGING TEST
        #mapping_file = f'{test_dir}/../LABEL-TO-BODY-mod-100-labelmap.csv'
        #block_stats_file = f'/tmp/block-statistics-testvol.h5'
        
        sys.argv += (f"--operation=indexes"
                     #f"--operation=sort-only"
                     #f"--operation=both"
                     #f" --agglomeration-mapping={mapping_file}"
                     f" --agglomeration-mapping={dvid_config['uuid']}"
                     f" --num-threads=8"
                     f" --batch-size=1000"
                     f" --tombstones=exclude"
                     f" --check-mismatches"
                     f" {dvid_config['server']}"
                     f" {dvid_config['uuid']}"
                     f" {dvid_config['segmentation-name']}"
                     f" {block_stats_file}".split())

    main()



