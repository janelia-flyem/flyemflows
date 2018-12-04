import os
import tempfile

import h5py
import numpy as np
import pandas as pd

import pytest
from ruamel.yaml import YAML
from flyemflows.bin.launchworkflow import launch_workflow

yaml = YAML()
yaml.default_flow_style = False

# Overridden below when running from __main__
CLUSTER_TYPE = os.environ.get('CLUSTER_TYPE', 'local-cluster')

@pytest.fixture(scope="module")
def setup_samplepoints():
    template_dir = tempfile.mkdtemp(suffix="samplepoints-template")
    
    # Create volume
    volume = np.random.randint(100, size=(128, 256, 512), dtype=np.uint64)
    volume_path = f"{template_dir}/volume.h5"
    with h5py.File(volume_path, 'w') as f:
        f['volume'] = volume
    
    num_points = 1000
    points_z = np.random.randint(volume.shape[0], size=num_points)
    points_y = np.random.randint(volume.shape[1], size=num_points)
    points_x = np.random.randint(volume.shape[2], size=num_points)
    
    points_df = pd.DataFrame({'z': points_z, 'y': points_y, 'x': points_x})
    points_df["extra"] = np.random.choice(['foo', 'bar'], num_points)
    
    # The workflow should be able to handle 'empty' bricks properly.
    # To test that, we'll remove points from a slab in the middle
    points_df = points_df.query('(z // 64) == 1')

    points_path = f'{template_dir}/input.csv'
    points_df.to_csv(points_path, index=False)

    config = {
        "workflow-name": "samplepoints",
        "cluster-type": CLUSTER_TYPE,
        
        "input": {
            "hdf5": {
                "path": volume_path,
                "dataset": "volume"
            },
            "geometry": {
                "message-block-shape": [64,64,256]
            },
        },
        
        "samplepoints": {
            "input-table": "input.csv" # relative path
        }
    }

    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    return template_dir, config, volume, points_df


def test_samplepoints(setup_samplepoints):
    template_dir, _config, volume, points_df = setup_samplepoints
    
    execution_dir, workflow = launch_workflow(template_dir, 1)
    
    final_config = workflow.config
    output_df = pd.read_csv(f'{execution_dir}/{final_config["samplepoints"]["output-table"]}')

    # Should appear in sorted order
    sorted_coords = points_df[['z', 'y', 'x']].sort_values(['z', 'y', 'x']).values
    assert (sorted_coords == output_df[['z', 'y', 'x']].values).all()
    
    labels = volume[(*sorted_coords.transpose(),)]
    assert (labels == output_df['label']).all()

    # 'extra' columns should be preserved, even
    # though they weren't used in the computation.
    input_extra = points_df.sort_values(['z', 'y', 'x'])['extra'].values
    output_extra = output_df.sort_values(['z', 'y', 'x'])['extra'].values
    assert (output_extra == input_extra).all(), "Extra column was not correctly preserved"


if __name__ == "__main__":
    CLUSTER_TYPE = os.environ['CLUSTER_TYPE'] = "synchronous"
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_samplepoints'])
