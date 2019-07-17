import os
import logging

import zarr
import numpy as np

from confiddler import validate
from neuclease.util import box_to_slicing

from ..util import replace_default_entries
from . import GeometrySchema, VolumeServiceReader, VolumeServiceWriter, SegmentationAdapters

logger = logging.getLogger(__name__)


ZarrServiceSchema = \
{
    "description": "Parameters specify an Zarr volume (a directory on the filesystem).\n"
                   "We always use zarr's NestedDirectoryStore, not the default DirectoryStore.\n",
    "type": "object",
    "required": ["path", "dataset"],

    "default": {},
    "properties": {
        "path": {
            "description": "Path of the .zarr directory, which may be a 'group' or an array",
            "type": "string",
            "minLength": 1
        },
        "dataset": {
            "description": "If the .zarr directory specified in 'path' is a zarr 'group',\n"
                           "specify the complete name of a dataset (array) within the zarr directory.\n",
            "type": "string",
            "default": ""
        },
        "dtype": {
            "description": "Datatype of the volume.  Must be specified when creating a new volume.",
            "type": "string",
            "enum": ["auto", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float32", "float64"],
            "default": "auto"
        },
        "writable": {
            "description": "Open the array in read/write mode.\n"
                           "If the directory (or dataset) doesn't exist yet, create it upon initialization.\n"
                           "(Requires an explicit dtype and bounding box.)\n"
                           "Note: \n"
                           "  It is not safe for multiple processes to write to the same brick simultaneously.\n"
                           "  Clients should ensure that each process is responsible for writing brick-aligned portions of the dataset.\n",
            "type": "boolean",
            "default": False
        }
    }
}

ZarrVolumeSchema = \
{
    "description": "Schema for an Zarr volume", # (for when a generic SegmentationVolumeSchema or GrayscaleVolumeSchema won't suffice)
    "type": "object",
    "default": {},
    #"additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
    "properties": {
        "zarr": ZarrServiceSchema,
        "geometry": GeometrySchema,
        "adapters": SegmentationAdapters
    }
}


DEFAULT_CHUNK_WIDTH = 64

class ZarrVolumeService(VolumeServiceReader, VolumeServiceWriter):
    """
    Note:
      It is not safe for multiple processes to write to the same brick simultaneously.
      Clients should ensure that each process is responsible for writing brick-aligned portions of the dataset.
    """
    def __init__(self, volume_config):
        validate(volume_config, ZarrVolumeSchema, inject_defaults=True)

        # Zarr settings
        path = volume_config["zarr"]["path"]
        dataset_name = volume_config["zarr"]["dataset"]
        dtype = volume_config["zarr"]["dtype"]
        writable = volume_config["zarr"]["writable"]

        if dataset_name == '/':
            dataset_name = ''

        # Geometry
        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        preferred_message_shape_zyx = np.array( volume_config["geometry"]["message-block-shape"][::-1] )
        block_width = volume_config["geometry"]["block-width"]
        assert list(volume_config["geometry"]["available-scales"]) == [0], \
            "ZarrVolumeService supports only scale 0"

        if not writable and not os.path.exists(path):
            raise RuntimeError(f"File does not exist: {path}\n"
                               "You did not specify 'writable' in the config, so I won't create it.:\n")

        if not writable and dataset_name and not os.path.exists(f"{path}/{dataset_name}"):
            raise RuntimeError(f"File does not exist: {path}\n"
                               "You did not specify 'writable' in the config, so I won't create it.:\n")

        mode = 'r'
        if writable:
            mode = 'a'
        
        if os.path.exists(f'{path}/{dataset_name}'):
            # If no dataset, then the path is direct to an array.
            if not dataset_name:
                self._group = None
                self._dataset = zarr.open(path, mode)
            else:
                #self._group = zarr.open(path, mode)
                store = zarr.NestedDirectoryStore(path)
                self._group = zarr.open(store=store, mode=mode)
                self._dataset = self._group[dataset_name]
        else:
            if not writable:
                raise RuntimeError(f"{path}/{dataset_name} does not exist.\n"
                                   "You did not specify 'writable' in the config, so I won't create it.\n")

            if dtype == "auto":
                raise RuntimeError(f"Can't create Zarr array {path}/{dataset_name}: No dtype specified in the config.")
            
            if -1 in bounding_box_zyx.flat:
                raise RuntimeError(f"Can't create Zarr array {path}/{dataset_name}: Bounding box is not completely specified in the config.")

            if block_width == -1:
                chunks = np.minimum(3*(DEFAULT_CHUNK_WIDTH,), bounding_box_zyx[1])
                replace_default_entries(chunks, 3*(DEFAULT_CHUNK_WIDTH,))
            else:
                chunks = 3*(block_width,)
            
            if not dataset_name:
                self._group = None
                self._dataset = zarr.open(path, mode)
            else:
                #self._group = zarr.open(path, mode)
                store = zarr.NestedDirectoryStore(path)
                self._group = zarr.open(store=store, mode=mode)
                self._dataset = self._group.create_dataset( dataset_name,
                                                            shape=bounding_box_zyx[1],
                                                            dtype=np.dtype(dtype),
                                                            chunks=tuple(chunks) )

        ###
        ### bounding_box_zyx
        ###
        replace_default_entries(bounding_box_zyx, [(0,0,0), self._dataset.shape])
        assert (bounding_box_zyx[0] >= 0).all()
        assert (bounding_box_zyx[1] <= self._dataset.shape).all(), \
            f"bounding box ({bounding_box_zyx.tolist()}) exceeds the stored zarr volume shape ({self._dataset.shape})"
        
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
        volume_config["zarr"]["dtype"] = self._dtype.name
        volume_config["geometry"]["block-width"] = chunk_shape[0]
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()
        

    @property
    def dtype(self):
        return self._dtype

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape_zyx

    @property
    def block_width(self):
        chunk_shape = self._dataset.chunks or self._dataset.shape
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
            ("Zarr volume service only supports scale 0 for now.\n"
             "As a workaround, try wrapping in a ScaledVolumeService by adding 'rescale-level: 0' to your 'adapters' config section.")
        return self._dataset[box_to_slicing(*box_zyx)]
    
    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        assert scale == 0
        
        box = np.array([offset_zyx, offset_zyx])
        box[1] += subvolume.shape

        self._dataset[box_to_slicing(*box)] = subvolume
