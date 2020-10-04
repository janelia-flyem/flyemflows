import pickle
import numpy as np
import pandas as pd

ORIG_STATS = 'stats_df.pkl'
FROM_SCALE = 1
MITO_CC_TABLE = '../03-mito-cc-20200930.073156/node_df_final.pkl'

print(f"Loading scale-{FROM_SCALE} stats")
with open(ORIG_STATS, 'rb') as f:
    stats_df = pickle.load(f)

stats_df = stats_df.rename(columns={
    'ellipsoid_radius_0': 'r0',
    'ellipsoid_radius_1': 'r1',
    'ellipsoid_radius_2': 'r2'})

# Drop unnecessary covariance/autocorrelation matrix elements
# Also drop the 'pca_radius' elements.
stats_df = stats_df[['z', 'y', 'x', 'total_size', 'class_1', 'class_2', 'class_3', 'r0', 'r1', 'r2']]

# Convert to scale-0
stats_df[[*'zyx', 'r0', 'r1', 'r2']] *= (2**FROM_SCALE)
stats_df[['total_size', 'class_1', 'class_2', 'class_3']] *= (2**FROM_SCALE)**3

# Compute eccentricities, cuz why not.
stats_df['eccentricity'] = np.sqrt(1 - stats_df.eval('r2**2 / r0**2'))

print("Loading CC table")
with open(MITO_CC_TABLE, 'rb') as f:
    cc_df = pickle.load(f)

# Note that identity-mappings are not included in this table,
# and every body has one mito that was identity-mapped
# (i.e. named with the same ID as the body).
print("Renaming CC columns")
mito_bodies = (cc_df[['final_cc', 'orig']].drop_duplicates()
                .rename(columns={'final_cc': 'mito_id', 'orig': 'body'})
                .set_index('mito_id'))

print("Merging CC to stats")
stats_df = stats_df.merge(mito_bodies, 'left', left_index=True, right_index=True)

mito_ids = stats_df.index
mito_ids = pd.Series(mito_ids, index=mito_ids)
stats_df['body'] = stats_df['body'].fillna(mito_ids).astype(np.uint64)

# Re-order columns
stats_df = stats_df[['body',
                     'x', 'y', 'z',
                     'total_size', 'class_1', 'class_2', 'class_3',
                     'r0', 'r1', 'r2', 'eccentricity']]

with open('final_stats_s0_df.pkl', 'wb') as f:
    pickle.dump(stats_df, f)
