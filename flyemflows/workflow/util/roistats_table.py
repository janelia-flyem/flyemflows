import warnings
import logging
from itertools import product
import pandas as pd

from neuclease.util import iter_batches, tqdm_proxy

logger = logging.getLogger(__name__)

def roistats_table(input_path,
                   neuprint_dataset,
                   voxel_col=None,
                   num_ranked=5,
                   exclude_none_roi=False,
                   neuprint_server='neuprint.janelia.org',
                   output_path=None):
    """
    The RoiStats workflow produces a list of body-ROI pairs and corresponding voxel counts.

    This function
        - appends fractional columns to the absolute numbers,
        - adds synapse counts (from neuprint),
        - filters for the top-N ROIs for each body,
        - and pivots (reshapes) the data so that the top-5 ROIs (and all stats)
          can be viewed on a single row of a spreadsheet.

    Args:
        input_path:
            CSV file produced by the RoiStats workflow, e.g. roi-stats.csv

        neuprint_dataset:
            Name of the neuprint dataset, e.v. 'hemibrain:v1.1'

        voxel_col:
            Name of the column which contains the voxel counts, e.g. voxels_s1

        num_ranked:
            How many ranked ROIs to include in the result columns

        exclude_none_roi:
            When computing totals (and fractions), should voxels that were not
            part of any ROI be excluded from the total (denomitator)?

        output_path:
            If provided, export the results to the given path.

        neuprint_server:
            Which neuprint server to use to obtain synapse counts.

    Returns:
        Table of top-N ROI stats for each body.
    """
    from neuprint import Client, fetch_neurons
    c = Client(neuprint_server, neuprint_dataset)

    stats = pd.read_csv(input_path)
    if voxel_col is None:
        vcs = [*filter(lambda c: c.startswith('voxels'), stats.columns)]
        if len(vcs) != 1:
            raise RuntimeError("Could not auto-determine voxel_col.  Please provide the column name to use.")
        voxel_col = vcs[0]

    stats = stats.rename(columns={f'{voxel_col}': f'roi_{voxel_col}'})

    if exclude_none_roi:
        # Exclude <none> from totals so fractions are
        # given in terms of neuropil totals only.
        stats = stats.query('roi_id != 0')

    # Compute totals and fractions
    voxel_totals = stats.groupby('body')[f'roi_{voxel_col}'].sum().rename(f'total_{voxel_col}')
    stats = stats.merge(voxel_totals, 'left', on='body')
    stats['roi_voxels_frac'] = stats.eval(f'roi_{voxel_col} / total_{voxel_col}')

    # Drop the '<none>' ROI, since that isn't conveniently available in the synapse
    # data, and probably isn't of interest anyway.
    # (Above, the '<none>' voxels were already counted in the totals (denominators)
    #  if necessary, but we never list '<none>' as an ROI column.)
    stats = stats.query('roi_id != 0').drop(columns=['roi_id'])

    # Fetch synapse counts from neuprint
    bodies = stats['body'].unique()
    logger.info(f"Fetching synapse counts from neuprint for {len(bodies)} bodies")
    neuron_batches, roi_syn_batches = [], []
    for bodies in tqdm_proxy(iter_batches(bodies, 10_000)):
        n, r = fetch_neurons(bodies, client=c)
        neuron_batches.append(n)
        roi_syn_batches.append(r)

    neurons = pd.concat(neuron_batches, ignore_index=True)
    roi_syn = pd.concat(roi_syn_batches, ignore_index=True)

    neurons = neurons.rename(columns={'bodyId': 'body'})
    roi_syn = roi_syn.rename(columns={'bodyId': 'body',
                                      'pre': 'roi_pre',
                                      'post': 'roi_post'})

    logger.info("Computing fractions")

    # Limit to primary (non-overlapping) rois
    roi_syn = roi_syn.query('roi in @c.primary_rois')

    # Extract totals and compute fractions
    syn_totals = (neurons[['body', 'pre', 'post']]
                    .rename(columns={'pre': 'total_pre',
                                     'post': 'total_post'}))

    roi_syn = roi_syn.merge(syn_totals, 'left', on='body')
    roi_syn['pre_frac'] = roi_syn.eval('roi_pre / total_pre')
    roi_syn['post_frac'] = roi_syn.eval('roi_post / total_post')

    # Combine voxel stats and synapse stats
    logger.info("Combining voxel stats and synapse stats")
    stats = stats.merge(roi_syn, 'left', on=['body', 'roi'])

    # Filter by top-N per body
    logger.info("Filtering by top-N ROIs per body")
    stats = stats.sort_values(['body', f'roi_{voxel_col}'], ascending=False)
    stats = stats.groupby('body').head(num_ranked)
    stats['roi_voxel_rank'] = stats.groupby('body').cumcount() + 1

    # Choose column order for viewing and
    # pivot (reshape) to put the top-N across columns
    logger.info("Pivoting table to place ROIs in columns")
    cols = ['roi', 'roi_voxels_frac', 'pre_frac', 'post_frac', f'roi_{voxel_col}', 'roi_post', 'roi_pre']
    stats_table = stats[['body', 'roi_voxel_rank', *cols]].pivot('body', 'roi_voxel_rank')
    stats_table = stats_table.swaplevel(axis=1)

    # Group by column type, then rank
    multicols = [(b,a) for (a,b) in product(cols, range(1, 1+num_ranked))]
    stats_table = stats_table.loc[:, multicols].fillna('')

    # Append totals
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"pandas\..*",
            message=".*merging between different levels can give an unintended result.*")
        stats_table = stats_table.merge(voxel_totals, 'left', left_index=True, right_index=True)
        stats_table = stats_table.merge(syn_totals.set_index('body'), 'left', left_index=True, right_index=True)

    # Append type/instance
    stats_table = stats_table.merge(neurons.set_index('body')[['type', 'instance']], 'left', left_index=True, right_index=True)

    # Set as index for aesthetics while viewing
    stats_table = stats_table.reset_index().set_index(['body', 'type', 'instance', f'total_{voxel_col}', 'total_pre', 'total_post'])
    stats_table.columns = pd.MultiIndex.from_tuples(stats_table.columns, names=['roi_voxel_rank', ''])

    if output_path:
        logger.info(f"Writing {output_path}")
        stats_table.to_csv(output_path)

    return stats_table


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-ranked-rois', '-r', type=int, default=5)
    parser.add_argument('--neuprint', default='neuprint.janelia.org')
    parser.add_argument('--voxel-col', help='Name of the input column that contains the voxel counts.')
    parser.add_argument('--exclude-none-roi', action='store_true')
    parser.add_argument('input_path')
    parser.add_argument('neuprint_dataset')
    parser.add_argument('output_path', nargs='?')
    args = parser.parse_args()

    if args.output_path is None:
        if args.exclude_none_roi:
            args.output_path = 'ranked-roi-table-excluding-non-neuropil.csv'
        else:
            args.output_path = 'ranked-roi-table.csv'

    from neuclease import configure_default_logging
    configure_default_logging()

    roistats_table(args.input_path,
                   args.neuprint_dataset,
                   args.voxel_col,
                   args.num_ranked_rois,
                   args.exclude_none_roi,
                   args.neuprint,
                   args.output_path)


if __name__ == "__main__":
    main()
