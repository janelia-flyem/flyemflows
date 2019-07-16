import os
import collections
from pathlib import Path

import z5py
import numpy as np

from .util import downsample

def export_to_multiscale_n5(volume, path, dataset_scale_0_name='s0', chunks=256, max_scale=0, downsample_method='subsample'):
    """
    Export the given ndarray to N5, creating a series of datasets (one for each scale).
    """
    if not isinstance(chunks, collections.Iterable):
        chunks = volume.ndim * (chunks,)
    
    assert len(chunks) == volume.ndim, "Chunk shape must correspond to volume dimensionality"
    
    if dataset_scale_0_name.startswith('/'):
        dataset_scale_0_name = dataset_scale_0_name[1:]
    
    assert dataset_scale_0_name.endswith('0') or max_scale == 0, \
        ("By convention, multiscale datasets will be suffixed with the scale number.\n"
        "Therefore, the scale-0 name must end with '0'.")

    assert 0 <= max_scale <= 9
    dataset_prefix = dataset_scale_0_name[:-1]

    p = Path(path) / dataset_scale_0_name
    if not p.parent.exists():
        os.makedirs(p.parent)
    
    f = z5py.File(path, use_zarr_format=False)
    for scale in range(max_scale+1):
        if scale > 0:
            volume = downsample(volume, 2, downsample_method)
            assert volume.flags.c_contiguous
        
        chunks = np.minimum( chunks, (volume.shape) ).tolist()
        ds = f.create_dataset(f'{dataset_prefix}{scale}', dtype=volume.dtype, shape=volume.shape, chunks=(*chunks,))
        ds[:] = volume
        assert (ds[:] == volume).all(), "z5py appears broken..."
