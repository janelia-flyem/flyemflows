import os
import logging
from pathlib import Path
from itertools import chain

import ujson
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease.util import read_csv_header, read_csv_col

logger = logging.getLogger(__name__)

# TODO: Might be nice to be able to specify 'top N' for a DVID labelmap instance...
BodyListSchema = {
    "description": "List of body IDs (or supervoxel IDs) to process, or a path to a CSV/JSON file with the list.\n",
    "oneOf": [
        {
            "description": "A list of body IDs (or supervoxel IDs).",
            "type": "array",
            "items": {"type": "integer"},
            "default": []
        },
        {
            "description": "A CSV file containing a single column of body IDs (or supervoxel IDs).",
            "type": "string",
            "default": ""
        }
    ],
    "default": []
}


def load_body_list(config_data, is_supervoxels):
    """
    Load a list of body IDs (or supervoxel IDs) from the given config data.
    The config data can be a list of body IDs (directly listed in the config),
    or a path to a CSV/feather file with a column of body IDs.

    If multiple columns are present, we select either the 'body' or 'sv' columns,
    depending on whether ``is_supervoxels`` is True.
    """
    if isinstance(config_data, list):
        return np.array(config_data, dtype=np.uint64)

    assert isinstance(config_data, str)

    bodies_path = config_data
    del config_data

    if is_supervoxels:
        col = 'sv'
    else:
        col = 'body'

    assert os.path.exists(bodies_path), \
        f"File does not exist: {bodies_path}"

    if bodies_path.endswith('.npy'):
        a = np.load(bodies_path)
        assert a.ndim == 1
        if a.dtype.names is None:
            # We assume that the user supplied the correct column (sv or body)
            return a.astype(np.uint64)
        if col not in a.dtype.names:
            msg = (
                f"File contains a structured array, but it does "
                f"not have a '{col}' column:\n{bodies_path}"
            )
            raise RuntimeError(msg)
        return a[col].astype(np.uint64)

    if bodies_path.endswith('.feather'):
        return feather.read_feather(bodies_path)[col].astype(np.uint64)

    header = read_csv_header(bodies_path)
    if header and col in header:
        bodies = pd.read_csv(bodies_path)[col].drop_duplicates()
    else:
        # Just read the first column, no matter what it's named
        logger.warning(f"No column named {col}, so reading first column instead")
        bodies = read_csv_col(bodies_path, 0, np.uint64).drop_duplicates()

    return bodies.values.astype(np.uint64)


LabelGroupSchema = {
    "description": "A specificaton for a set of label groups, specified as\n"
                   "either a CSV file with 'label' and 'group' columns\n"
                   "or a JSON file structured as a list-of-lists.\n"
                   "You may also provide a list-of-lists directly in this config field.\n",
    "oneOf": [
        {
            "description": "A list of body IDs (or supervoxel IDs) to generate meshes for.",
            "type": "array",
            "items": {"type": "array", "items": { "type": "integer" }},
            "default": []
        },
        {
            "description": "Either a CSV/feather file with 'label' and 'group' columns\n"
                           "or a JSON file structured as a list-of-lists\n",
            "type": "string",
            "default": ""
        }
    ],
    "default": [],
}


def load_label_groups(config_data):
    """
    Load the given config data (see ``LabelGroupSchema``),
    and return a DataFrame with columns ['label', 'group']
    """
    if isinstance(config_data, list):
        groups = config_data
        return _groups_to_df(groups, 'config data')

    assert isinstance(config_data, str)
    path = Path(config_data)
    assert path.suffix in ('.json', '.csv', '.feather')

    if path.suffix == '.csv':
        df = pd.read_csv(path, dtype=np.uint64)
        if not (set(df.columns) >= set(['label', 'group'])):
            msg = f"Label group CSV file does not have the expected columns 'label' and 'group':\n{path}"
            raise RuntimeError(msg)
        df = df[['label', 'group']].drop_duplicates().reset_index(drop=True)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            groups = ujson.load(f)
        df = _groups_to_df(groups, path)
    else:
        df = feather.read_feather(path)

    if df.max().max() <= np.iinfo(np.uint32).max:
        df = df.astype(np.uint32)

    if not (set(df.columns) & {'label', 'group'}) and (set(df.columns) >= {'body', 'sv'}):
        df = df.rename(columns={'sv': 'label', 'body': 'group'})[['label', 'group']]

    return df


def _groups_to_df(groups, path):
    assert isinstance(groups, list), \
        f"Label group JSON does not have the correct structure in:\n {path}"
    assert all( isinstance(group, list) for group in groups ), \
        f"Label group JSON does not have the correct structure in:\n {path}"

    labels = np.fromiter(chain(*groups), np.uint64)
    lens = list(map(len, groups))
    flag_pos = np.add.accumulate(lens[:-1])
    start_flags = np.zeros(len(labels), np.uint32)
    if len(start_flags) > 0:
        start_flags[flag_pos] = 1
    group_ids = 1+np.add.accumulate(start_flags, dtype=np.uint32)

    df = pd.DataFrame({'label': labels, 'group': group_ids})
    return df.drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    _groups = [[1,2], [3,4], [4,5,6]]
    _groups_df = load_label_groups(_groups)
    print(_groups_df)
