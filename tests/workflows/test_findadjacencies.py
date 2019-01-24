import os
import tempfile

import h5py
import numpy as np
import pandas as pd

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchflow import launch_flow

yaml = YAML()
yaml.default_flow_style = False

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')

@pytest.fixture(scope="module")
def setup_findadjacencies():
    template_dir = tempfile.mkdtemp(suffix="findadjacencies-template")
    
    # Create volume
    volume = np.zeros((256,256,256), np.uint64)
    volume[30,  0:192, 0:64] = 1
    volume[31,  0:192, 0:64] = 2

    volume[140,  0:192, 0:64] = 4
    volume[141,  0:192, 0:64] = 3
    
    volume_path = f"{template_dir}/volume.h5"
    with h5py.File(volume_path, 'w') as f:
        f['volume'] = volume
    
    config = {
        "workflow-name": "findadjacencies",
        "cluster-type": CLUSTER_TYPE,
        
        "input": {
            "hdf5": {
                "path": volume_path,
                "dataset": "volume"
            },
            "geometry": {
                "message-block-shape": [64,64,64]
            },
        },
        
        "findadjacencies": {
            "output-table": "output.csv"
        }
    }

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    return template_dir, config, volume


def test_findadjacencies(setup_findadjacencies):
    template_dir, _config, _volume = setup_findadjacencies
    
    execution_dir, workflow = launch_flow(template_dir, 1)
    
    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["findadjacencies"]["output-table"]}')

    #print(output_df.columns)
    #print(output_df)

    label_pairs = list(map(tuple, output_df[['label_a', 'label_b']].values))
    assert (1,2) in label_pairs
    assert (3,4) in label_pairs
    
    assert output_df.query('label_a == 1')['z'].iloc[0] == 30
    assert output_df.query('label_a == 1')['forwardness'].iloc[0] == True

    assert output_df.query('label_a == 3')['z'].iloc[0] == 140
    assert output_df.query('label_a == 3')['forwardness'].iloc[0] == False


if __name__ == "__main__":
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_findadjacencies'])
