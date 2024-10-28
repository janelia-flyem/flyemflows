import os
import json
import platform

import zarr
import numcodecs
import numpy as np
from skimage.util import view_as_blocks

from confiddler import validate, flow_style
from neuclease.util import box_to_slicing, choose_pyramid_depth, box_intersection, boxes_from_grid, ndrange, dump_json

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
            "minimum": -1,
            "maxValue": 10, # arbitrary limit, but if you're using a higher value, you're probably mistaken.
            "default": -1
        },
        "compression": {
            "description": "What type of compression to use.  We only support 'gzip' and options supported by numcodecs.Blosc\n",
            "type": "string",
            "enum": ['', 'gzip', 'blosc-blosclz', 'blosc-lz4', 'blosc-lz4hc', 'blosc-snappy', 'blosc-zlib', 'blosc-zstd'],
            "default": 'blosc-zstd'
        },
        "resolution": {
            "description": "Resolution of the s0 data, in nanometers",
            "type": "array",
            "items": { "type": "number" },
            "minItems": 3,
            "maxItems": 3,
            "default": flow_style([8.0, 8.0, 8.0])
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
                           "In general, you should use NestedDirectoryStore if your volume is large.\n"
                           'Choices: ["DirectoryStore", "NestedDirectoryStore", "N5Store", "ZipStore", "DBMStore", "LMDBStore", "SQLiteStore", "FSStore"]',
            "type": "string",
            "enum": ["DirectoryStore", "NestedDirectoryStore", "N5Store", "ZipStore", "DBMStore", "LMDBStore", "SQLiteStore", "FSStore"],
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
        "out-of-bounds-access": {
            "description": "If 'forbid', any out-of-bounds read/write is an error.\n"
                           "If 'permit', out-of-bounds reads are permitted, and filled with zeros, and out-of-bounds writes are merely ignored.\n"
                           "If 'permit-empty', out-of-bounds writes are permitted (and ignored) only if the out-of-bounds voxels are zero.\n"
                           "Otherwise, they're an error.",
            "type": "string",
            "enum": ["forbid", "permit", "permit-empty"],
            "default": "permit-empty"
        },
        "write-empty-blocks": {
            "description": "Whether to write blocks which are completely zeros.  \n"
                           "For new datasets, it's best to avoid the writes.  If you're overwriting an existing dataset,\n"
                           "then using this setting is not advisable unless you know that no empty blocks will need to\n"
                           "be written to erase old data.",
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

        self._store_cls = getattr(zarr, volume_config["zarr"]["store-type"])
        self._zarr_file = None
        self._zarr_datasets = {}

        self._ensure_datasets_exist(volume_config)
        self._ensure_multires_n5_attributes(volume_config)

        if isinstance(self.zarr_dataset(0), zarr.hierarchy.Group):
            raise RuntimeError("The Zarr dataset you specified appears to be a 'group', not a volume.\n"
                               "Please pass the complete dataset name.  If your dataset is multi-scale,\n"
                               "pass the name of scale 0 as the dataset name (e.g. 's0').\n")

        chunk_shape = np.array(self.zarr_dataset(0).chunks)
        assert len(chunk_shape) == 3

        ##
        ## message-block-shape
        ##
        preferred_message_shape_zyx = np.array(volume_config["geometry"]["message-block-shape"])[::-1]

        # Replace -1's in the message-block-shape with the corresponding chunk_shape dimensions.
        replace_default_entries(preferred_message_shape_zyx, chunk_shape)
        missing_shape_dims = (preferred_message_shape_zyx == -1)
        preferred_message_shape_zyx[missing_shape_dims] = chunk_shape[missing_shape_dims]

        if (preferred_message_shape_zyx % chunk_shape).any():
            msg = (f"zarr volume: Expected message-block-shape ({preferred_message_shape_zyx[::-1]}) "
                  f"to be a multiple of the chunk shape ({chunk_shape[::-1]})")
            logger.warning(msg)

        ##
        ## message-grid-offset
        ##
        preferred_grid_offset_zyx = np.array( volume_config["geometry"]["message-grid-offset"][::-1] )

        # The notion of 'block-width' doesn't really make sense if the chunks aren't cubes,
        # but we'll assume the user has chosen something reasonable and just use the minimum chunk dimension.
        block_width = min(chunk_shape)

        global_offset = np.array(volume_config["zarr"]["global-offset"][::-1])
        uncropped_bounding_box_zyx = np.array([(0,0,0), self.zarr_dataset(0).shape])
        uncropped_bounding_box_zyx += global_offset

        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        assert (uncropped_bounding_box_zyx[1] >= bounding_box_zyx[1]).all() or volume_config["zarr"]["out-of-bounds-access"] != "forbid", \
            f"Volume config bounding box ({bounding_box_zyx}) exceeds the bounding box of the data ({uncropped_bounding_box_zyx}).\n"\
            f"If you want to enable reading out-of-bounds regions (as empty), add out-of-bounds-access: 'permit-empty' to your config."

        # Replace -1 bounds with auto
        missing_bounds = (bounding_box_zyx == -1)
        bounding_box_zyx[missing_bounds] = uncropped_bounding_box_zyx[missing_bounds]

        # Store members
        self._uncropped_bounding_box_zyx = uncropped_bounding_box_zyx
        self._bounding_box_zyx = bounding_box_zyx
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._preferred_grid_offset_zyx = preferred_grid_offset_zyx
        self._block_width = block_width
        self._available_scales = volume_config["geometry"]["available-scales"]
        self._global_offset = global_offset
        self._out_of_bounds_access = volume_config["zarr"]["out-of-bounds-access"]

        # Overwrite config entries that we might have modified
        volume_config["geometry"]["block-width"] = int(self._block_width)
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()
        volume_config["geometry"]["message-grid-offset"] = self._preferred_grid_offset_zyx[::-1].tolist()

    @property
    def dtype(self):
        return self.zarr_dataset(0).dtype

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape_zyx

    @property
    def preferred_grid_offset(self):
        return self._preferred_grid_offset_zyx

    @property
    def block_width(self):
        return self._block_width

    @property
    def uncropped_bounding_box_zyx(self):
        return self._uncropped_bounding_box_zyx

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
        msg = f"Zarr Request is out-of-bounds (scale={scale}) (XYZ): {orig_box[:, ::-1].tolist()}"
        if self._out_of_bounds_access in ("permit", "permit-empty"):
            logger.warning(msg)
        else:
            msg += "\nAdd 'out-of-bounds-access' to your config to allow such requests"
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
            self._write_subvolume(subvolume, box, scale)
            return

        msg = (f"Box extends beyond Zarr scale {scale} volume bounds (XYZ): "
               f"{box[:, ::-1].tolist()} exceeds {stored_bounding_box[:, ::-1].tolist()}")

        if self._out_of_bounds_access == 'forbid':
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
        if self._out_of_bounds_access == 'permit-empty' and subvol_copy.any():
            # Note that this message shows the true zarr storage bounds,
            # and doesn't show the logical bounds according to global_offset (if any).
            msg = (f"Cannot write subvolume. Box extends beyond Zarr scale {scale} volume storage bounds (XYZ): "
                f"{box[:, ::-1].tolist()} exceeds {stored_bounding_box[:, ::-1].tolist()}\n"
                "and the out-of-bounds portion is not empty (contains non-zero values).\n")
            raise RuntimeError(msg)

        logger.warning(msg)
        clipped_subvolume = subvolume[box_to_slicing(*clipped_box - box[0])]
        self._write_subvolume(clipped_subvolume, clipped_box, scale)

    def _write_subvolume(self, subvolume, box, scale):
        assert (subvolume.shape == box[1] - box[0]).all(), \
            f"shape (XYZ) {subvolume.shape[::-1]} doesn't match box (XYZ) {box[:, ::-1].tolist()} (scale={scale})"

        if self._write_empty_blocks:
            self.zarr_dataset(scale)[box_to_slicing(*box)] = subvolume
            return

        # Check each block to see if any are empty
        block_shape = self.zarr_dataset(scale).chunks
        block_boxes = boxes_from_grid(box, block_shape, clipped=True)

        if not (box % block_shape).any():
            # Faster path for block-aligned subvolumes
            subvolume = np.asarray(subvolume, order='C')
            block_vols = view_as_blocks(subvolume, block_shape)
            block_flags = block_vols.any(axis=(3,4,5)).ravel()
        else:
            block_flags = []
            for block_box in block_boxes:
                has_data = subvolume[box_to_slicing(*block_box - box[0])].any()
                block_flags.append(has_data)

        if all(block_flags):
            # No empty blocks; write it all in one operation
            self.zarr_dataset(scale)[box_to_slicing(*box)] = subvolume
            return

        if not any(block_flags):
            # All blocks are empty
            return

        # At least one empty block;
        # So write blocks individually, skipping the empty ones.
        for block_box, has_data in zip(block_boxes, block_flags):
            if has_data:
                self.zarr_dataset(scale)[box_to_slicing(*block_box)] = subvolume[box_to_slicing(*block_box - box[0])]


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
        if compression == 'gzip':
            compressor = numcodecs.GZip()
        elif compression.startswith('blosc-'):
            cname = compression[len('blosc-'):]
            compressor = numcodecs.Blosc(cname)
        else:
            assert compression == "", f"Unimplemented compression: {compression}"

        self._write_empty_blocks = volume_config["zarr"]["write-empty-blocks"]

        if create_if_necessary:
            max_scale = volume_config["zarr"]["creation-settings"]["max-scale"]
            if max_scale == -1:
                if -1 in creation_shape:
                    raise RuntimeError("Can't auto-determine the appropriate max-scale to create "
                                       "(or extend) the data with, because you didn't specify a "
                                       "volume creation shape (or bounding box")
                max_scale = choose_pyramid_depth(creation_shape, 512)

                # Overwrite config
                volume_config["zarr"]["creation-settings"]["max-scale"] = max_scale

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

    def _ensure_multires_n5_attributes(self, volume_config):
        """
        If we're writing to n5, then the parent directory containing
        s0, s1, etc. should have an attributes.json file with certain values
        that neuroglancer (and BigDataViewer) knows how to interpret.
        """
        if volume_config["zarr"]["store-type"] != "N5Store":
            return

        writable = volume_config["zarr"]["writable"]
        if writable is None:
            writable = volume_config["zarr"]["create-if-necessary"]

        if not writable:
            return

        max_scale = volume_config["zarr"]["creation-settings"]["max-scale"]
        assert max_scale >= 0
        scales = [[2**s, 2**s, 2**s] for s in range(1+max_scale)]

        new_attributes = {
            "pixelResolution": {
                "dimensions": volume_config["zarr"]["creation-settings"]["resolution"],
                "unit":"nm"
            },
            "multiScale": True,
            "scales": scales,
            "axes":["x", "y", "z"],
            "units":["nm", "nm", "nm"],
            #"n5":"2.5.0",
            #"translate":[0, 0, 0]
        }

        path = volume_config["zarr"]["path"]
        dset_name = volume_config["zarr"]["dataset"]
        dset_path = f'{path}/{dset_name}'
        dset_dir = os.path.dirname(dset_path)
        attrs_path = f'{dset_dir}/attributes.json'

        # Combine with existing attributes (if any)
        with open(attrs_path, 'r') as f:
            attributes = json.load(f)

        attributes.update(new_attributes)
        attributes.setdefault('n5', '2.5.0')

        dump_json(attributes, attrs_path, unsplit_int_lists=True)
