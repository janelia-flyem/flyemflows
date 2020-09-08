"""
Example post-processing script for the RoiStats workflow.

If you process all of the 'leaf' ROIs (i.e. the ROIs at the bottom of the hierarchy),
and those leaf ROIs perfectly "tile" their parent ROIs, then it is possible to
run RoiStats for just the leaf ROIs, and then add stats for the non-leaf ROIs by
summing the leaf ROI totals under each compound ROI.

This script was used to generate the complete ROI stats for the hemibrain.
Original data is (or was) here:

    /groups/flyem/data/scratchspace/flyemflows/roistats/traced-roi-stats-20200902.122056/
"""
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from neuprint import Client, fetch_roi_hierarchy

print("Reading ROI hierarchy")
c = Client('neuprint.janelia.org', 'hemibrain:v1.1')
g = fetch_roi_hierarchy(include_subprimary=True, mark_primary=False, format='nx')

# These hemibrain ROIs overlap with other ROIs,
# so they aren't truly part of the hierarchy.
# Ignore them. (They were also ignored when the roi-stats were computed.)
bad_rois = {'dACA(R)', 'lACA(R)', 'vACA(R)'}

# Produce a map of each roi to its bottom-level component rois.
print("Computing roi leaf descendants")
leaf_descendants = {}
for r in g.nodes():
    if r not in bad_rois:
        des = {*nx.descendants(g, r)} | {r}
        des -= bad_rois
        des = filter(lambda d: g.out_degree(d) == 0, des)
        leaf_descendants[r] = sorted(des)

# Load the roi-stats for leaf rois (from the cluster job)
print("Loading leaf roi-stats")
df = pd.DataFrame(np.load('roi-stats.npy', allow_pickle=True))

# Pivot so roi names are in the columns
print("Pivoting stats")
pdf = df[['body', 'roi', 'voxels_s1']].pivot('body', 'roi')
pdf = pdf.fillna(0).astype(np.int64)
pdf.columns = pdf.columns.droplevel()

# Compute non-leaf roi stats by summing the proper leaf rois for each
print("Computing non-leaf stats")
for roi in tqdm(leaf_descendants.keys()):
    if roi not in pdf.columns:
        ld = leaf_descendants[roi]
        pdf[roi] = pdf.loc[:, ld].sum(axis=1)

# Sort columns by name
pdf = pdf.sort_index(axis=1)

# Save as npy and csv
p = 'roi-stats-pivoted-plus-nonleaf-rois.npy'
print(f"Saving {p}")
np.save(p, pdf.to_records(index=True))

p = 'roi-stats-pivoted-plus-nonleaf-rois.csv'
print(f"Saving {p}")
pdf.to_csv('roi-stats-pivoted-plus-nonleaf-rois.csv', index=True, header=True)

print("DONE")
