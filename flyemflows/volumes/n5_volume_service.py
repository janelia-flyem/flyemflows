import os
import platform

import z5py
import numpy as np

from confiddler import validate, flow_style
from neuclease.util import box_to_slicing, choose_pyramid_depth

from ..util import replace_default_entries
from . import VolumeServiceWriter, GeometrySchema, GrayscaleAdapters

import logging
logger = logging.getLogger(__name__)


N5CreationSettingsSchema = \
{
    "description": "Settings to use when creating an N5 volume.\n",
    "type": "object",
    "default": {},
    "properties": {
        "shape": {
            "description": "The shape of the volume.\n"
                           "If not provided, it is automatically set from the bounding-box upper coordinate.\n",
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 3,
            "maxItems": 3,
            "default": flow_style([-1,-1,-1])
        },
        "dtype": {
            "description": "Datatype of the volume.  Must be specified when creating a new volume.",
            "type": "string",
            "enum": ["auto", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float32", "float64"],
            "default": "auto"
        },
        "block-shape": {
            "desription": "The shape of the blocks on disk.",
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 3,
            "maxItems": 3,
            "default": flow_style([128,128,128])
        },
        "max-scale": {
            "description": "How many additional subdirectories to create for multi-scale volumes.\n"
                           "If unset (-1), then a default max-scale will be chosen automatically \n "
                           "based on a heuristic involving the volume shape.\n",
            "type": "integer",
            "minValue": -1,
            "maxValue": 10, # arbitrary limit, but if you're using a higher value, you're probably mistaken.
            "default": -1
        },
        "compression": {
            "description": "The type of compression to use for all blocks.\n"
                           'Choices: ["raw", "gzip", "bzip2", "xz"]\n'
                           "Note: At the time of this writing, neuroglancer only supports 'raw' and 'gzip'.\n"
                           "Note: In theory, 'blosc' and 'lz4' should be allowed, but z5py doesn't support those for n5.\n",
            "type": "string",
            "enum": ["raw", "gzip", "bzip2", "xz", "lz4"], # don't be fooled -- blosc is allowed by z5py, but not for N5
            "default": "raw"
        },
        "compression-level": {
            "description": "The compression setting (ignored when compression is 'raw').",
            "type": "integer",
            "default": 5
        }
    }
}


N5ServiceSchema = \
{
    "description": "Parameters to specify an N5 volume (or set of multiscale volumes)",
    "type": "object",
    "required": ["path", "dataset"],
    "default": {},
    
    "properties": {
        "path": {
            "description": "Path to the n5 parent directory, which may contain multiple datasets",
            "type": "string",
            "minLength": 1
        },
        "dataset": {
            "description": "Name of the volume.\n"
                           "If the volume is stored at multiple scales,\n"
                           "then by convention the scale must be included as a suffix on each volume name.\n"
                           "In this config, please list the scale-0 name, e.g. 'my-grayscale0', or '22-34/s0', etc.\n",
            "type": "string",
            "minLength": 1
        },
        "writable": {
            "description": "If True, open the array in read/write mode, otherwise open in read-only mode.\n"
                           "By default, guess based on create-if-necessary.\n",
            "oneOf": [ {"type": "boolean"}, {"type": "null"} ],
            "default": None
        },
        "create-if-necessary": {
            "description": "Whether or not to create the array directory on disk if it doesn't already exist.\n"
                           "If you expect the array to exist on the server already, leave this\n"
                           "set to False to avoid confusion in the case of typos, etc.\n",
            "type": "boolean",
            "default": False
        },
        "creation-settings": N5CreationSettingsSchema,
    }
}

N5VolumeSchema = \
{
    "description": "Describes a volume from N5.\n"
                   "Note: \n"
                   "  It is not safe for multiple processes to write to the same block simultaneously.\n"
                   "  Clients should ensure that each process is responsible for writing brick-aligned portions of the dataset.\n",
    "type": "object",
    "default": {},
    "properties": {
        "n5": N5ServiceSchema,
        "geometry": GeometrySchema,
        "adapters": GrayscaleAdapters
    }
}

class N5VolumeService(VolumeServiceWriter):

    def __init__(self, volume_config):
        validate(volume_config, N5VolumeSchema, inject_defaults=True)
        
        # Convert path to absolute if necessary (and write back to the config)
        path = os.path.abspath(volume_config["n5"]["path"])
        self._path = path
        volume_config["n5"]["path"] = self._path

        dataset_name = volume_config["n5"]["dataset"]
        self._dataset_name = dataset_name
        if self._dataset_name.startswith('/'):
            self._dataset_name = self._dataset_name[1:]
        volume_config["n5"]["dataset"] = self._dataset_name

        self._n5_file = None
        self._n5_datasets = {}
        
        self._ensure_datasets_exist(volume_config)

        if isinstance(self.n5_dataset(0), z5py.group.Group):
            raise RuntimeError("The N5 dataset you specified appears to be a 'group', not a volume.\n"
                               "Please pass the complete dataset name.  If your dataset is multi-scale,\n"
                               "pass the name of scale 0 as the dataset name (e.g. 's0').\n")

        chunk_shape = np.array(self.n5_dataset(0).chunks)
        assert len(chunk_shape) == 3

        preferred_message_shape_zyx = np.array(volume_config["geometry"]["message-block-shape"])[::-1]

        # Replace -1's in the message-block-shape with the corresponding chunk_shape dimensions.
        replace_default_entries(preferred_message_shape_zyx, chunk_shape)
        missing_shape_dims = (preferred_message_shape_zyx == -1)
        preferred_message_shape_zyx[missing_shape_dims] = chunk_shape[missing_shape_dims]
        assert not (preferred_message_shape_zyx % chunk_shape).any(), \
            f"Expected message-block-shape ({preferred_message_shape_zyx}) "\
            f"to be a multiple of the chunk shape ({chunk_shape})"

        if chunk_shape[0] == chunk_shape[1] == chunk_shape[2]:
            block_width = int(chunk_shape[0])
        else:
            # The notion of 'block-width' doesn't really make sense if the chunks aren't cubes.
            block_width = -1
        
        auto_bb = np.array([(0,0,0), self.n5_dataset(0).shape])

        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        assert (auto_bb[1] >= bounding_box_zyx[1]).all(), \
            f"Volume config bounding box ({bounding_box_zyx}) exceeds the bounding box of the data ({auto_bb})."

        # Replace -1 bounds with auto
        missing_bounds = (bounding_box_zyx == -1)
        bounding_box_zyx[missing_bounds] = auto_bb[missing_bounds]

        # Store members
        self._bounding_box_zyx = bounding_box_zyx
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._block_width = block_width
        self._available_scales = volume_config["geometry"]["available-scales"]

        # Overwrite config entries that we might have modified
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()

    @property
    def dtype(self):
        return self.n5_dataset(0).dtype

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape_zyx

    @property
    def block_width(self):
        return self._block_width
    
    @property
    def bounding_box_zyx(self):
        return self._bounding_box_zyx

    @property
    def available_scales(self):
        return self._available_scales

    def get_subvolume(self, box_zyx, scale=0):
        box_zyx = np.asarray(box_zyx)
        return self.n5_dataset(scale)[box_to_slicing(*box_zyx.tolist())]
    

    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        offset_zyx = np.asarray(offset_zyx)
        box = np.array([offset_zyx, offset_zyx+subvolume.shape])
        self.n5_dataset(scale)[box_to_slicing(*box)] = subvolume


    @property
    def n5_file(self):
        # This member is memoized because that makes it
        # easier to support pickling/unpickling.
        if self._n5_file is None:
            need_permissions_fix = not os.path.exists(self._path)
            self._n5_file = z5py.File(self._path, self._filemode)

            if need_permissions_fix:
                # Set default permissions to be group-writable
                if platform.system() == "Linux":
                    os.system(f"chmod g+rw {self._path}")
                    os.system(f'setfacl -d -m g::rw {self._path}')
                elif platform.system() == "Darwin":
                    pass # FIXME: I don't know how to do this on macOS.

        return self._n5_file


    def n5_dataset(self, scale):
        if scale not in self._n5_datasets:
            if scale == 0:
                name = self._dataset_name
            else:
                assert 0 <= scale < 10 # need to fix the logic below if you want to support higher scales
                assert self._dataset_name[-1] == '0', \
                    "The N5 dataset does not appear to be a multi-resolution dataset."
                name = self._dataset_name[:-1] + f'{scale}'

            self._n5_datasets[scale] = self.n5_file[name]

        return self._n5_datasets[scale]


    def _ensure_datasets_exist(self, volume_config):
        dtype = volume_config["n5"]["creation-settings"]["dtype"]
        create_if_necessary = volume_config["n5"]["create-if-necessary"]
        writable = volume_config["n5"]["writable"]
        if writable is None:
            writable = create_if_necessary
        
        mode = 'r'
        if writable:
            mode = 'a'
        self._filemode = mode

        block_shape = volume_config["n5"]["creation-settings"]["block-shape"]

        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        creation_shape = np.array(volume_config["n5"]["creation-settings"]["shape"][::-1])
        replace_default_entries(creation_shape, bounding_box_zyx[1])
        
        compression = volume_config["n5"]["creation-settings"]["compression"]
        compression_options = {}
        if compression != "raw":
            compression_options['level'] = volume_config["n5"]["creation-settings"]["compression-level"]

        if create_if_necessary:
            max_scale = volume_config["n5"]["creation-settings"]["max-scale"]
            if max_scale == -1:
                if -1 in creation_shape:
                    raise RuntimeError("Can't auto-determine the appropriate max-scale to create "
                                       "(or extend) the data with, because you didn't specify a "
                                       "volume creation shape (or bounding box")
                max_scale = choose_pyramid_depth(creation_shape, 512)
            
            available_scales = [*range(1+max_scale)]
        else:
            available_scales = volume_config["geometry"]["available-scales"]

            if not os.path.exists(self._path):
                raise RuntimeError(f"File does not exist: {self._path}\n"
                                   "You did not specify 'writable' in the config, so I won't create it.:\n")

            if self._dataset_name and not os.path.exists(f"{self._path}/{self._dataset_name}"):
                raise RuntimeError(f"File does not exist: {self._path}/{self._dataset_name}\n"
                                   "You did not specify 'writable' in the config, so I won't create it.:\n")

        for scale in available_scales:
            if scale == 0:
                name = self._dataset_name
            else:
                name = self._dataset_name[:-1] + f'{scale}'

            if name not in self.n5_file:
                if not writable:
                    raise RuntimeError(f"Dataset for scale {scale} does not exist, and you "
                                       "didn't specify 'writable' in the config, so I won't create it.")

                if dtype == "auto":
                    raise RuntimeError(f"Can't create N5 array {self._path}/{self._dataset_name}: "
                                       "No dtype specified in the config.")

                # Use 128 if the user didn't specify a chunkshape
                replace_default_entries(block_shape, 3*[128])

                # z5py complains if the chunks are larger than the shape,
                # which could happen here if we aren't careful (for higher scales).
                scaled_shape = (creation_shape // (2**scale))
                chunks = np.minimum(scaled_shape, block_shape).tolist()
                if (chunks != block_shape) and (scale == 0):
                    logger.warning(f"Block shape ({block_shape}) is too small for "
                                   f"the dataset shape ({creation_shape}). Shrinking block shape.")
                
                self._n5_datasets[scale] = self.n5_file.create_dataset( name,
                                                                        scaled_shape.tolist(),
                                                                        np.dtype(dtype),
                                                                        chunks=chunks,
                                                                        compression=compression,
                                                                        **compression_options )

    def __getstate__(self):
        """
        Pickle representation.
        """
        d = self.__dict__.copy()
        # Don't attempt to pickle the underlying C++ objects
        # Instead, set them to None so it will be lazily regenerated after unpickling.
        d['_n5_file'] = None
        d['_n5_datasets'] = {}
        return d
