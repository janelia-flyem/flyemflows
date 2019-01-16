import os
import logging

import numpy as np
import pandas as pd

from neuclease.util import read_csv_header, read_csv_col

logger = logging.getLogger(__name__)

BodyListSchema = {
    "description": "List of body IDs (or supervoxel IDs) to process, or a path to a CSV file with the list.",
    "oneOf": [
        {
            "description": "A list of body IDs (or supervoxel IDs) to generate meshes for.",
            "type": "array",
            "default": []
        },
        {
            "description": "A CSV file containing a single column of body IDs (or supervoxel IDs) to generate meshes for.",
            "type": "string",
            "default": ""
        }
    ],
    "default": []
}

def load_body_list(config_data, is_supervoxels):
    if isinstance(config_data, list):
        return config_data

    bodies_csv = config_data
    del config_data

    assert os.path.exists(bodies_csv), \
        f"CSV file does not exist: {bodies_csv}"
        
    if is_supervoxels:
        col = 'sv'
    else:
        col = 'body'
    
    if col in read_csv_header(bodies_csv):
        bodies = pd.read_csv(bodies_csv)[col].drop_duplicates()
    else:
        # Just read the first column, no matter what it's named
        logger.warning(f"No column named {col}, so reading first column instead")
        bodies = read_csv_col(bodies_csv, 0, np.uint64).drop_duplicates()

    return bodies
