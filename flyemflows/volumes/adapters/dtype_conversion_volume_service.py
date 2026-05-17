import logging

import numpy as np

from .. import VolumeServiceWriter

logger = logging.getLogger(__name__)


_VALID_DTYPES = [
    "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64",
    "float32", "float64",
]


DtypeConversionSchema = \
{
    "description": "Convert data to a different dtype during reads and/or writes.\n"
                   "Useful for narrowing oversized integer labels (e.g. uint64 -> uint32)\n"
                   "without rewriting the underlying volume, or for transparently bridging\n"
                   "between an underlying integer volume and float-typed consumers.\n",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "dtype": {
            "description": "Target dtype that callers see.  Leave empty ('') to disable this adapter.\n",
            "type": "string",
            "enum": ["", *_VALID_DTYPES],
            "default": "",
        },
        "apply-when": {
            "description": "When to apply the conversion.\n"
                           "  reading             - cast read data from the underlying dtype to the target dtype.\n"
                           "  writing             - cast user-supplied data from the target dtype to the underlying dtype.\n"
                           "  reading-and-writing - both.\n",
            "type": "string",
            "enum": ["reading", "writing", "reading-and-writing"],
            "default": "reading-and-writing",
        },
        "overflow-mode": {
            "description": "What to do if a value cannot be represented exactly in the destination dtype.\n"
                           "  permit - silently truncate/wrap (numpy's default cast behavior).\n"
                           "  error  - raise if any value would lose information.\n",
            "type": "string",
            "enum": ["permit", "error"],
            "default": "error",
        },
    },
}


class DtypeConversionVolumeService(VolumeServiceWriter):
    """
    Wraps an existing VolumeServiceReader/Writer and presents a view of it in
    a different dtype, casting transparently on each read and/or write.

    The user-facing dtype reported by this service is always the configured
    target dtype, regardless of apply-when -- callers should configure
    apply-when to match how they intend to use the service.
    """

    def __init__(self, original_volume_service, dtype_config):
        self.original_volume_service = original_volume_service
        target = dtype_config["dtype"]
        if not target:
            raise RuntimeError(
                "DtypeConversionVolumeService requires a non-empty 'dtype'. "
                "Omit the adapter entirely if you don't want any conversion."
            )
        self._target_dtype = np.dtype(target)
        self._source_dtype = np.dtype(original_volume_service.dtype)
        self.apply_when_reading = dtype_config["apply-when"] in ("reading", "reading-and-writing")
        self.apply_when_writing = dtype_config["apply-when"] in ("writing", "reading-and-writing")
        self._overflow_mode = dtype_config["overflow-mode"]

    @property
    def base_service(self):
        return self.original_volume_service.base_service

    @property
    def dtype(self):
        return self._target_dtype

    @property
    def block_width(self):
        return self.original_volume_service.block_width

    @property
    def preferred_message_shape(self):
        return self.original_volume_service.preferred_message_shape

    @property
    def preferred_grid_offset(self):
        return self.original_volume_service.preferred_grid_offset

    @property
    def uncropped_bounding_box_zyx(self):
        return self.original_volume_service.uncropped_bounding_box_zyx

    @property
    def bounding_box_zyx(self):
        return self.original_volume_service.bounding_box_zyx

    @property
    def available_scales(self):
        return self.original_volume_service.available_scales

    def get_subvolume(self, box_zyx, scale=0):
        volume = self.original_volume_service.get_subvolume(box_zyx, scale)
        if self.apply_when_reading and volume.dtype != self._target_dtype:
            volume = self._cast(volume, self._target_dtype)
        return volume

    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        if self.apply_when_writing and subvolume.dtype != self._source_dtype:
            subvolume = self._cast(subvolume, self._source_dtype)
        self.original_volume_service.write_subvolume(subvolume, offset_zyx, scale)

    def _cast(self, array, target_dtype):
        converted = array.astype(target_dtype)
        if self._overflow_mode == "error":
            # Round-trip back to the source dtype; if any value differs, the cast was lossy.
            roundtrip = converted.astype(array.dtype)
            if not np.array_equal(roundtrip, array):
                raise RuntimeError(
                    f"DtypeConversionVolumeService: value out of range while casting "
                    f"{array.dtype} -> {target_dtype}"
                )
        return converted
