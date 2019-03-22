import os
import logging

import h5py
import numpy as np

from confiddler import validate
from neuclease.util import box_to_slicing

from ..util import replace_default_entries
from . import GeometrySchema, VolumeServiceReader, VolumeServiceWriter, SegmentationAdapters

logger = logging.getLogger(__name__)


Hdf5ServiceSchema = \
{
    "description": "Parameters specify an HDF5 volume (on the filesystem).",
    "type": "object",
    "required": ["path", "dataset"],

    "default": {},
    "properties": {
        "path": {
            "description": "Path of the .h5 file",
            "type": "string",
            "minLength": 1
        },
        "dataset": {
            "description": "Complete name of the dataset with in the file, including any groups.",
            "type": "string",
            "minLength": 1
        },
        "dtype": {
            "description": "Datatype of the volume.  Must be specified when creating a new volume.",
            "type": "string",
            "enum": ["auto", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float32", "float64"],
            "default": "auto"
        },
        "writable": {
            "description": "Open the file in read/write mode.\n"
                           "If the file (or dataset) doesn't exist yet, create it upon initialization.\n"
                           "(Requires an explicit dtype and bounding box.)\n"
                           "Note: \n"
                           "  Writability is not currently safe in multi-process mode.\n"
                           "  Writing HDF5 volumes is only to be used for testing (in a single process).\n",
            "type": "boolean",
            "default": False
        }
    }
}

Hdf5VolumeSchema = \
{
    "description": "Schema for an HDF5 volume", # (for when a generic SegmentationVolumeSchema or GrayscaleVolumeSchema won't suffice)
    "type": "object",
    "default": {},
    #"additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "hdf5": Hdf5ServiceSchema,
        "geometry": GeometrySchema,
        "adapters": SegmentationAdapters
    }
}


DEFAULT_CHUNK_WIDTH = 64

class Hdf5VolumeService(VolumeServiceReader, VolumeServiceWriter):
    """
    Note: Writability is not currently safe in multi-process mode.
          Writing HDF5 volumes via this class is suitable only for testing (in a single process).
    """

    def __init__(self, volume_config):
        validate(volume_config, Hdf5VolumeSchema, inject_defaults=True)

        # HDF5 settings
        path = volume_config["hdf5"]["path"]
        dataset_name = volume_config["hdf5"]["dataset"]
        dtype = volume_config["hdf5"]["dtype"]
        writable = volume_config["hdf5"]["writable"]

        # Geometry
        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        preferred_message_shape_zyx = np.array( volume_config["geometry"]["message-block-shape"][::-1] )
        block_width = volume_config["geometry"]["block-width"]
        assert list(volume_config["geometry"]["available-scales"]) == [0], \
            "Hdf5VolumeService supports only scale 0"

        if not writable and not os.path.exists(path):
            raise RuntimeError(f"File does not exist: {path}\n"
                               "You did not specify 'writable' in the config, so I won't create it.:\n")

        mode = 'r'
        if writable:
            mode = 'a'
        
        self._h5_file = h5py.File(path, mode)

        if dataset_name in self._h5_file:
            self._dataset = self._h5_file[dataset_name]
        else:
            if not writable:
                raise RuntimeError(f"Dataset '{dataset_name}' not found in file: {path}\n"
                                   "You did not specify 'writable' in the config, so I won't create it.\n")

            if dtype == "auto":
                raise RuntimeError(f"Can't create dataset '{dataset_name}': No dtype specified in the config.")
            
            if -1 in bounding_box_zyx.flat:
                raise RuntimeError(f"Can't create dataset '{dataset_name}': Bounding box is not completely specified in the config.")

            if block_width == -1:
                chunks = np.minimum(3*(DEFAULT_CHUNK_WIDTH,), bounding_box_zyx[1])
                replace_default_entries(chunks, 3*(DEFAULT_CHUNK_WIDTH,))
            else:
                chunks = 3*(block_width,)
            
            self._dataset = self._h5_file.create_dataset( dataset_name,
                                                          shape=bounding_box_zyx[1],
                                                          dtype=np.dtype(dtype),
                                                          chunks=tuple(chunks) )

        ###
        ### bounding_box_zyx
        ###
        replace_default_entries(bounding_box_zyx, [(0,0,0), self._dataset.shape])
        assert (bounding_box_zyx[0] >= 0).all()
        assert (bounding_box_zyx[1] <= self._dataset.shape).all(), \
            f"bounding box ({bounding_box_zyx.tolist()}) exceeds the stored hdf5 volume shape ({self._dataset.shape})"
        
        ###
        ### dtype
        ###
        dtype = self._dataset.dtype

        ###
        ### preferred_message_shape_zyx
        ###
        chunk_shape = self._dataset.chunks or self._dataset.shape
        assert len(self._dataset.shape) == 3, f"Dataset '{dataset_name} isn't 3D"
        if -1 in preferred_message_shape_zyx:
            assert (preferred_message_shape_zyx == -1).all(), \
                "Please specify the entire message shape in your config (or omit it entirely)"

            # Aim for bricks of ~256 MB
            MB = 2**20
            chunk_bytes = np.prod(chunk_shape) * dtype.itemsize
            chunks_per_brick = max(1, 256*MB // chunk_bytes)
            preferred_message_shape_zyx = np.array((*chunk_shape[:2], chunk_shape[2]*chunks_per_brick))
        
        if block_width == -1:
            block_width = chunk_shape[0]
        else:
            assert block_width == chunk_shape[0], \
                "block-width does not match file chunk shape"

        ##
        ## Store members
        ##
        self._mode = mode
        self._path = path
        self._dataset_name = dataset_name
        self._bounding_box_zyx = bounding_box_zyx
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._dtype = self._dataset.dtype

        ##
        ## Overwrite config entries that we might have modified
        ##
        volume_config["hdf5"]["dtype"] = self._dtype.name
        volume_config["geometry"]["block-width"] = chunk_shape[0]
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()
        

    def __getstate__(self):
        """
        Pickle representation.
        """
        d = self.__dict__.copy()
        # These attributes are not pickleable.
        # Set them to None so it will be lazily regenerated after unpickling.
        d['_h5_file'] = None
        d['_dataset'] = None
        return d

    @property
    def h5_file(self):
        # See __getstate__()
        if self._h5_file is None:
            self._h5_file = h5py.File(self._path, self._mode)        
        return self._h5_file

    @property
    def dataset(self):
        # See __getstate__()
        if self._dataset is None:
            self._dataset = self.h5_file[self._dataset_name]        
        return self._dataset

    @property
    def dtype(self):
        return self._dtype

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape_zyx

    @property
    def block_width(self):
        chunk_shape = self.dataset.chunks or self.dataset.shape
        assert (chunk_shape[0] == chunk_shape[1] == chunk_shape[2]), \
            ("Dataset chunks are not isotropic. block-width shouldn't be used.")
        return chunk_shape[0]
    
    @property
    def bounding_box_zyx(self):
        return self._bounding_box_zyx

    @property
    def available_scales(self):
        return [0]
    
    def get_subvolume(self, box_zyx, scale=0):
        assert scale == 0, \
            ("Hdf5 volume service only supports scale 0 for now.]\n"
             "As a workaround, try wrapping in a ScaledVolumeService by adding 'rescale-level: 0' to your 'adapters' config section.")
        return self.dataset[box_to_slicing(*box_zyx)]
    
    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        assert scale == 0
        
        box = np.array([offset_zyx, offset_zyx])
        box[1] += subvolume.shape

        self.dataset[box_to_slicing(*box)] = subvolume
