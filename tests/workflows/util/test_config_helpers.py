import tempfile

import numpy as np
import pandas as pd

import pytest

from flyemflows.workflow.util.config_helpers import load_label_groups


def test_load_label_groups_from_lists():
    groups = [[1,2], [3,4], [4,5,6]]
    groups_df = load_label_groups(groups)
    assert groups_df.columns.tolist() == ['label', 'group']
    assert groups_df.values.tolist() == [[1,1],
                                         [2,1],
                                         [3,2],
                                         [4,2],
                                         [4,3],
                                         [5,3],
                                         [6,3]]

def test_load_label_groups_from_csv():
    groups = [1,1,2,2,3,3,3]
    labels = [1,2,3,4,4,5,6]
    df = pd.DataFrame([*zip(groups, labels)], columns=['group', 'label'])
    df['unused-column'] = 555

    d = tempfile.mkdtemp()
    path = f"{d}/label-groups.csv"
    df.to_csv(path)
    
    groups_df = load_label_groups(path)
    assert groups_df.columns.tolist() == ['label', 'group']
    assert groups_df.dtypes.to_dict() == {'label': np.uint64, 'group': np.uint64}
    assert groups_df.values.tolist() == [[1,1],
                                         [2,1],
                                         [3,2],
                                         [4,2],
                                         [4,3],
                                         [5,3],
                                         [6,3]]

if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.util.test_config_helpers'])
