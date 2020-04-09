from itertools import product
import pandas as pd

NEUPRINT_SERVER = 'neuprint-test.janelia.org'

# Produced by the RoiStats workflow
ROISTATS_CSV = 'roi-stats.csv'

# The top N ROIs will be shown
NUM_RANKED = 5

# If ROI totals (and fractions) should not account for the <none> ROI
EXCLUDE_NONE_ROI = False
if EXCLUDE_NONE_ROI:
    OUTPUT_PATH = 'ranked-roi-table-excluding-non-neuropil.csv'
else:
    OUTPUT_PATH = 'ranked-roi-table.csv'

# RoiStats workflow names the column with the scale
VOXEL_COL = 'voxels_s1'


def roistats_table(input_path=ROISTATS_CSV, voxel_col=VOXEL_COL, output_path=OUTPUT_PATH):
    """
    The RoiStats workflow produces a list of body-ROI pairs and corresponding voxel counts.

    This function
        - appends fractional columns to the absolute numbers,
        - adds synapse counts (from neuprint),
        - filters for the top-N ROIs for each body,
        - and pivots (reshapes) the data so that the top-5 ROIs (and all stats)
          can be viewed on a single row of a spreadsheet.

    Note:
        Edit the globals above as needed before running this script.
    """
    from neuprint import Client, fetch_neurons
    c = Client(NEUPRINT_SERVER, 'hemibrain')

    stats = pd.read_csv(input_path)
    stats = stats.rename(columns={f'{voxel_col}': f'roi_{voxel_col}'})

    if EXCLUDE_NONE_ROI:
        # Exclude <none> from totals so fractions are
        # given in terms of neuropil totals only.
        stats = stats.query('roi_id != 0')

    # Compute totals and fractions
    voxel_totals = stats.groupby('body')[f'roi_{voxel_col}'].sum().rename(f'total_{voxel_col}')
    stats = stats.merge(voxel_totals, 'left', on='body')
    stats['roi_voxels_frac'] = stats.eval(f'roi_{voxel_col} / total_{voxel_col}')

    # Drop the '<none>' ROI, since that isn't conveniently available in the synapse
    # data, and probably isn't of interest anyway.
    stats = stats.query('roi_id != 0').drop(columns=['roi_id'])

    # Fetch synapse counts from neuprint
    neurons, roi_syn = fetch_neurons(stats['body'].values, client=c)

    neurons = neurons.rename(columns={'bodyId': 'body'})
    roi_syn = roi_syn.rename(columns={'bodyId': 'body',
                                      'pre': 'roi_pre',
                                      'post': 'roi_post'})

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
    stats = stats.merge(roi_syn, 'left', on=['body', 'roi'])

    # Filter by top-N per body
    stats = stats.sort_values(['body', f'roi_{voxel_col}'], ascending=False)
    stats = stats.groupby('body').head(NUM_RANKED)
    stats['roi_voxel_rank'] = stats.groupby('body').cumcount() + 1

    # Choose column order for viewing and
    # pivot (reshape) to put the top-N across columns
    cols = ['roi', 'roi_voxels_frac', 'pre_frac', 'post_frac', f'roi_{voxel_col}', 'roi_post', 'roi_pre']
    stats_table = stats[['body', 'roi_voxel_rank', *cols]].pivot('body', 'roi_voxel_rank')
    stats_table = stats_table.swaplevel(axis=1)

    # Group by column type, then rank
    multicols = [(b,a) for (a,b) in product(cols, range(1, 1+NUM_RANKED))]
    stats_table = stats_table.loc[:, multicols].fillna('')

    # Add totals
    stats_table = stats_table.merge(voxel_totals, 'left', left_index=True, right_index=True)
    stats_table = stats_table.merge(syn_totals.set_index('body'), 'left', left_index=True, right_index=True)

    # Add type/instance
    stats_table = stats_table.merge(neurons.set_index('body')[['type', 'instance']], 'left', left_index=True, right_index=True)

    # Set as index for aesthetics while viewing
    stats_table = stats_table.reset_index().set_index(['body', 'type', 'instance', f'total_{voxel_col}', 'total_pre', 'total_post'])
    stats_table.columns = pd.MultiIndex.from_tuples(stats_table.columns, names=['roi_voxel_rank', ''])

    if output_path:
        stats_table.to_csv(output_path)

    return stats_table


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('voxel_col')
    parser.add_argument('output_path', nargs='?', default=OUTPUT_PATH)
    args = parser.parse_args()

    roistats_table(args.input_path, args.voxel_col, args.output_path)
