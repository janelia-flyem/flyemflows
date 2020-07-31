import os
import platform

import zarr
import numcodecs
import numpy as np

from confiddler import validate, flow_style
from neuclease.util import box_to_slicing, choose_pyramid_depth, box_intersection

from ..util import replace_default_entries
from . import VolumeServiceWriter, GeometrySchema, GrayscaleAdapters

import logging
logger = logging.getLogger(__name__)


ZarrCreationSettingsSchema = \
{
    "description": "Settings to use when creating an Zarr volume.\n",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "shape": {
            "description": "The shape of the volume.\n"
                           "If not provided, it is automatically set from the bounding-box upper coordinate and global-offset (if any).\n",
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
        "chunk-shape": {
            "desription": "The shape of the chunks on disk.",
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
            "description": "What type of compression to use.  We only use options supported by numcodecs.Blosc\n",
            "type": "string",
            "enum": ['', 'blosc-blosclz', 'blosc-lz4', 'blosc-lz4hc', 'blosc-snappy', 'blosc-zlib', 'blosc-zstd'],
            "default": 'blosc-zstd'
        }
    }
}


ZarrServiceSchema = \
{
    "description": "Parameters to specify an Zarr volume (or set of multiscale volumes)",
    "type": "object",
    "required": ["path", "dataset"],
    "default": {},
    "additionalProperties": False,
    "properties": {
        "path": {
            "description": "Path to the zarr parent directory, which may contain multiple datasets",
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
        "store-type": {
            # https://github.com/zarr-developers/zarr-python/issues/530
            "description": "Zarr supports an assortment of 'store' types, and unfortunately it's\n"
                           "impossible to infer the type of store used from the zarr metadata.\n"
                           "We must specify which type of store to use when opening the volume.\n"
                           "In general, you should use NestedDirectoryStore if your volume is large.\n",
            "type": "string",
            "enum": ["DirectoryStore", "NestedDirectoryStore"],
            "default": "NestedDirectoryStore"
        },
        "global-offset": {
            "description": "Indicates what global coordinate corresponds to item (0,0,0) of the zarr array.\n"
                           "This offset will be subtracted from all read/write requests before accessing the zarr container.\n"
                           "Pass this offset in [X,Y,Z] order!\n"
                           "Note: If your offset is not a multiple of the chunk shape, then some workflows may not work as efficiently.\n",
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3,
            "default": flow_style([0,0,0])
        },
        "permit-out-of-bounds": {
            "description": "If true, out-of-bounds reads will be permitted.  Out-of-bounds voxels are returned as 0.\n"
                           "Out-of-bounds writes will also be permitted, as long as the out-of-bounds portion is entirely 0.\n"
                           "Otherwise, out-of-bounds writes are treated as an error.\n",
            "type": "boolean",
            "default": True
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
        "creation-settings": ZarrCreationSettingsSchema,

    }
}

ZarrVolumeSchema = \
{
    "description": "Describes a volume from Zarr.\n"
                   "Note: \n"
                   "  It is not safe for multiple processes to write to the same block simultaneously.\n"
                   "  Clients should ensure that each process is responsible for writing brick-aligned portions of the dataset.\n",
    "type": "object",
    "default": {},
    "properties": {
        "zarr": ZarrServiceSchema,
        "geometry": GeometrySchema,
        "adapters": GrayscaleAdapters
    }
}

class ZarrVolumeService(VolumeServiceWriter):

    def __init__(self, volume_config):
        validate(volume_config, ZarrVolumeSchema, inject_defaults=True)

        # Convert path to absolute if necessary (and write back to the config)
        path = os.path.abspath(volume_config["zarr"]["path"])
        self._path = path
        volume_config["zarr"]["path"] = self._path

        dataset_name = volume_config["zarr"]["dataset"]
        self._dataset_name = dataset_name
        if self._dataset_name.startswith('/'):
            self._dataset_name = self._dataset_name[1:]
        volume_config["zarr"]["dataset"] = self._dataset_name

        store_cfg = volume_config["zarr"]["store-type"]
        self._store_cls = { 'DirectoryStore': zarr.storage.DirectoryStore,
                            'NestedDirectoryStore': zarr.storage.NestedDirectoryStore
                          }[store_cfg]

        self._zarr_file = None
        self._zarr_datasets = {}

        self._ensure_datasets_exist(volume_config)

        if isinstance(self.zarr_dataset(0), zarr.hierarchy.Group):
            raise RuntimeError("The Zarr dataset you specified appears to be a 'group', not a volume.\n"
                               "Please pass the complete dataset name.  If your dataset is multi-scale,\n"
                               "pass the name of scale 0 as the dataset name (e.g. 's0').\n")

        chunk_shape = np.array(self.zarr_dataset(0).chunks)
        assert len(chunk_shape) == 3

        preferred_message_shape_zyx = np.array(volume_config["geometry"]["message-block-shape"])[::-1]

        # Replace -1's in the message-block-shape with the corresponding chunk_shape dimensions.
        replace_default_entries(preferred_message_shape_zyx, chunk_shape)
        missing_shape_dims = (preferred_message_shape_zyx == -1)
        preferred_message_shape_zyx[missing_shape_dims] = chunk_shape[missing_shape_dims]

        if (preferred_message_shape_zyx % chunk_shape).any():
            msg = (f"zarr volume: Expected message-block-shape ({preferred_message_shape_zyx[::-1]}) "
                  f"to be a multiple of the chunk shape ({chunk_shape[::-1]})")
            logger.warning(msg)

        if chunk_shape[0] == chunk_shape[1] == chunk_shape[2]:
            block_width = int(chunk_shape[0])
        else:
            # The notion of 'block-width' doesn't really make sense if the chunks aren't cubes.
            block_width = -1

        global_offset = np.array(volume_config["zarr"]["global-offset"][::-1])
        auto_bb = np.array([(0,0,0), self.zarr_dataset(0).shape])
        auto_bb += global_offset

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
        self._global_offset = global_offset
        self._permit_out_of_bounds = volume_config["zarr"]["permit-out-of-bounds"]

        # Overwrite config entries that we might have modified
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()

    @property
    def dtype(self):
        return self.zarr_dataset(0).dtype

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
        box_zyx = np.array(box_zyx)
        orig_box = box_zyx.copy()
        box_zyx -= (self._global_offset // (2**scale))

        clipped_box = box_intersection(box_zyx, [(0,0,0), self.zarr_dataset(scale).shape])
        if (clipped_box == box_zyx).all():
            return self.zarr_dataset(scale)[box_to_slicing(*box_zyx.tolist())]

        # Note that this message shows the true zarr storage bounds,
        # and doesn't show the logical bounds according to global_offset (if any).
        msg = f"Zarr Request is out-of-bounds (XYZ): {orig_box[:, ::-1].tolist()}"
        if self._permit_out_of_bounds:
            logger.warning(msg)
        else:
            msg += "\nAdd permit-out-of-bounds to your config to allow such requests"
            raise RuntimeError(msg)

        if (clipped_box[1] - clipped_box[0] <= 0).any():
            # request is completely out-of-bounds; just return zeros
            return np.zeros(box_zyx[1] - box_zyx[0], self.dtype)

        # Request is partially out-of-bounds; read what we can, zero-fill for the rest.
        clipped_vol = self.zarr_dataset(scale)[box_to_slicing(*clipped_box.tolist())]
        result = np.zeros(box_zyx[1] - box_zyx[0], self.dtype)
        localbox = clipped_box - box_zyx[0]
        result[box_to_slicing(*localbox)] = clipped_vol
        return result

    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        offset_zyx = np.array(offset_zyx)
        offset_zyx -= self._global_offset // (2**scale)
        box = np.array([offset_zyx, offset_zyx+subvolume.shape])

        stored_bounding_box = (self._bounding_box_zyx - self._global_offset) // (2**scale)
        if (box[0] >= 0).all() and (box[1] <= stored_bounding_box[1]).all():
            # Box is fully contained within the Zarr volume bounding box.
            self.zarr_dataset(scale)[box_to_slicing(*box)] = subvolume
        else:
            msg = ("Box extends beyond Zarr volume bounds (XYZ): "
                   f"{box[:, ::-1].tolist()} exceeds {stored_bounding_box[:, ::-1].tolist()}")

            if self._permit_out_of_bounds:
                logger.warning(msg)
            else:
                # Note that this message shows the true zarr storage bounds,
                # and doesn't show the logical bounds according to global_offset (if any).
                msg = "Cannot write subvolume. " + msg
                msg += "\nAdd permit-out-of-bounds to your config to allow such writes,"
                msg += " assuming the out-of-bounds portion is completely empty."
                raise RuntimeError(msg)

        clipped_box = box_intersection(box, stored_bounding_box)

        # If any of the out-of-bounds portion is non-empty, that's an error.
        subvol_copy = subvolume.copy()
        subvol_copy[box_to_slicing(*(clipped_box - box[0]))] = 0
        if subvol_copy.any():
            # Note that this message shows the true zarr storage bounds,
            # and doesn't show the logical bounds according to global_offset (if any).
            msg = ("Cannot write subvolume. Box extends beyond Zarr volume storage bounds (XYZ): "
                   f"{box[:, ::-1].tolist()} exceeds {stored_bounding_box[:, ::-1].tolist()}\n"
                   "and the out-of-bounds portion is not empty (contains non-zero values).\n")
            raise RuntimeError(msg)

        clipped_subvolume = subvolume[box_to_slicing(*clipped_box - box[0])]
        self.zarr_dataset(scale)[box_to_slicing(*clipped_box)] = clipped_subvolume


    @property
    def zarr_file(self):
        # This member is memoized because that makes it
        # easier to support pickling/unpickling.
        if self._zarr_file is None:
            need_permissions_fix = not os.path.exists(self._path)
            store = self._store_cls(self._path)
            self._zarr_file = zarr.open(store=store, mode=self._filemode)

            if need_permissions_fix:
                # Set default permissions to be group-writable
                if platform.system() == "Linux":
                    os.system(f"chmod g+rw {self._path}")
                    os.system(f'setfacl -d -m g::rw {self._path}')
                elif platform.system() == "Darwin":
                    pass # FIXME: I don't know how to do this on macOS.

        return self._zarr_file


    def zarr_dataset(self, scale):
        if scale not in self._zarr_datasets:
            if scale == 0:
                name = self._dataset_name
            else:
                assert 0 <= scale < 10 # need to fix the logic below if you want to support higher scales
                assert self._dataset_name[-1] == '0', \
                    "The Zarr dataset does not appear to be a multi-resolution dataset."
                name = self._dataset_name[:-1] + f'{scale}'

            self._zarr_datasets[scale] = self.zarr_file[name]

        return self._zarr_datasets[scale]


    def _ensure_datasets_exist(self, volume_config):
        dtype = volume_config["zarr"]["creation-settings"]["dtype"]
        create_if_necessary = volume_config["zarr"]["create-if-necessary"]
        writable = volume_config["zarr"]["writable"]
        if writable is None:
            writable = create_if_necessary

        mode = 'r'
        if writable:
            mode = 'a'
        self._filemode = mode

        block_shape = volume_config["zarr"]["creation-settings"]["chunk-shape"][::-1]

        global_offset = volume_config["zarr"]["global-offset"][::-1]
        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        creation_shape = np.array(volume_config["zarr"]["creation-settings"]["shape"][::-1])
        replace_default_entries(creation_shape, bounding_box_zyx[1] - global_offset)

        compression = volume_config["zarr"]["creation-settings"]["compression"]
        if compression:
            assert compression.startswith('blosc-')
            cname = compression[len('blosc-'):]
            compressor = numcodecs.Blosc(cname)
        else:
            compressor = None

        if create_if_necessary:
            max_scale = volume_config["zarr"]["creation-settings"]["max-scale"]
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
                                   "You did not specify 'create-if-necessary' in the config, so I won't create it.:\n")

            if self._dataset_name and not os.path.exists(f"{self._path}/{self._dataset_name}"):
                raise RuntimeError(f"File does not exist: {self._path}/{self._dataset_name}\n"
                                   "You did not specify 'create-if-necessary' in the config, so I won't create it.:\n")

        for scale in available_scales:
            if scale == 0:
                name = self._dataset_name
            else:
                name = self._dataset_name[:-1] + f'{scale}'

            if name not in self.zarr_file:
                if not writable:
                    raise RuntimeError(f"Dataset for scale {scale} does not exist, and you "
                                       "didn't specify 'writable' in the config, so I won't create it.")

                if dtype == "auto":
                    raise RuntimeError(f"Can't create Zarr array {self._path}/{self._dataset_name}: "
                                       "No dtype specified in the config.")

                # Use 128 if the user didn't specify a chunkshape
                replace_default_entries(block_shape, 3*[128])

                # zarr misbehaves if the chunks are larger than the shape,
                # which could happen here if we aren't careful (for higher scales).
                scaled_shape = (creation_shape // (2**scale))
                chunks = np.minimum(scaled_shape, block_shape).tolist()
                if (chunks != block_shape) and (scale == 0):
                    logger.warning(f"Block shape ({block_shape}) is too small for "
                                   f"the dataset shape ({creation_shape}). Shrinking block shape.")

                self._zarr_datasets[scale] = self.zarr_file.create_dataset( name,
                                                                            shape=scaled_shape.tolist(),
                                                                            dtype=np.dtype(dtype),
                                                                            chunks=chunks,
                                                                            compressor=compressor )
