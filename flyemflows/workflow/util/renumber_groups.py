import os
import numpy as np
import pandas as pd

from neuclease.util import tqdm_proxy

def renumber_groups(tabs_and_paths, output_dir, exclude_labels=[]):
    """
    Given a series of tab-wise label group CSVs,
    renumber the group IDs so that group IDs are not duplicated across tabs.
    """
    tables = []
    for tab, path in tqdm_proxy(tabs_and_paths.items()):
        tab_df = pd.read_csv(path, dtype=np.int64)
        assert tab_df.columns.tolist() == ['group', 'label']
        tab_df['tab'] = tab
        tables.append(tab_df)
        
    full_df = pd.concat(tables, ignore_index=True)
    full_df = full_df.query('label not in @exclude_labels')

    new_groups_df = full_df.drop_duplicates(['group', 'tab']).reset_index(drop=True)
    new_groups_df.index.name = 'unique_group'
    new_groups_df = new_groups_df.reset_index()

    full_regrouped_df = full_df.merge(new_groups_df[['tab', 'group', 'unique_group']], 'left', on=['tab', 'group'])

    full_regrouped_df = full_regrouped_df.drop(columns=['group']).rename(columns={'unique_group': 'group'})
    full_regrouped_df['group'] += 1

    os.makedirs(output_dir, exist_ok=True)
    for tab, tab_df in tqdm_proxy(full_regrouped_df.groupby('tab'), total=len(tabs_and_paths)):
        tab_df[['group', 'label']].to_csv(f'{output_dir}/renumbered-groups-tab{tab}.csv', header=True, index=False)

if __name__ == "__main__":
    tabs_and_paths = {}
    for tab in range(22,35):
        tabs_and_paths[tab] = f'tables/label-groups-tab{tab}-psd1.csv'

    bad_bodies_path = '/nrs/flyem/bergs/complete-ffn-agglo/bad-bodies-2019-02-26.csv'
    bad_bodies = pd.read_csv(bad_bodies_path)['body']
    renumber_groups(tabs_and_paths, 'renumbered-tables', bad_bodies)

