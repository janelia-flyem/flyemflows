import tempfile
import functools

import h5py
import numpy as np
import pandas as pd

import pytest
from ruamel.yaml import YAML
from jsonschema import ValidationError
from flyemflows.bin.launchworkflow import launch_workflow


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

    points_path = f'{template_dir}/input.csv'
    points_df.to_csv(points_path, index=False)

    config = {
        "workflow-name": "samplepoints",
        "cluster-type": "synchronous", # DEBUG
        
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

    yaml = YAML()
    yaml.default_flow_style = False
    with open(f"{template_dir}/workflow.yaml", 'w') as f:
        yaml.dump(config, f)

    return template_dir, config, volume, points_df


def print_validation_errors(f):
    """
    Decorator.
    
    Apparently pytest has trouble handling ValidationErrors for some reason,
    so this decorator ensures that we at least see them on the console before pytest chokes.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as ex:
            print(ex)
            raise

    return wrapper


@print_validation_errors
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


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.workflows.test_samplepoints'])
