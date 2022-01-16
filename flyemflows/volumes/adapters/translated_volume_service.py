import logging
import numpy as np

from .. import VolumeServiceReader

from confiddler import flow_style

logger = logging.getLogger(__name__)

TranslateSchema = \
{
    "description": "Translate the volume.",
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": { "type": "integer" },
    "default": flow_style([0, 0, 0])  # no translation
}


class TranslatedVolumeService(VolumeServiceReader):
    """
    Wraps an existing VolumeServiceReader and presents
    a translated view of it.

    Note: Since VolumeServices have no notion of an offset block grid,
          The translated service's grid will not necessarily be aligned to the original block grid.
          If the translation is not block-aligned, then requests that appear block-aligned in the
          translated volume will not actually be block-aligned with respect to the underlying data.
          Depending on your use-case, that might be desirable, or might not.
    """

    def __init__(self, original_volume_service, translation_xyz=[0, 0, 0]):
        """
        """
        assert len(translation_xyz) == 3
        translation_zyx = np.asarray(translation_xyz[::-1])
        del translation_xyz
        assert np.issubdtype(translation_zyx.dtype, np.integer), "translation dtype must be integer"
        self.translation_zyx = translation_zyx
        self.original_volume_service = original_volume_service

    @property
    def base_service(self):
        return self.original_volume_service.base_service

    @property
    def dtype(self):
        return self.original_volume_service.dtype

    @property
    def block_width(self):
        return self.original_volume_service.block_width

    @property
    def preferred_message_shape(self):
        return self.original_volume_service.preferred_message_shape

    @property
    def bounding_box_zyx(self):
        return self.original_volume_service.bounding_box_zyx + self.translation_zyx

    @property
    def available_scales(self):
        return self.original_volume_service.available_scales

    def get_subvolume(self, box_zyx, scale=0):
        scaled_translation = self.translation_zyx // 2**scale
        box_zyx = np.array(box_zyx)
        box_zyx -= scaled_translation
        return self.original_volume_service.get_subvolume(box_zyx, scale)

    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        scaled_translation = self.translation_zyx // 2**scale
        offset_zyx = np.array(offset_zyx)
        offset_zyx -= scaled_translation
        self.original_volume_service.write_subvolume(subvolume, offset_zyx, scale)
