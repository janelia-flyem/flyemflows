import os
import lz4.frame
import numpy as np

from dvidutils import LabelMapper
from confiddler import validate

from .labelmap_utils import LabelMapSchema, load_labelmap
from .. import VolumeServiceWriter


class LabelmappedVolumeService(VolumeServiceWriter):
    """
    Wraps an existing VolumeServiceReader/Writer for label data
    and presents a view of it in which all values are remapped
    according to a given mapping.
    
    (Technically, this is an example of the so-called
    "decorator" GoF pattern.)
    
    Note: This class uses only one mapping. It is valid to apply the same mapping
          for both reading and writing, provided that the mapping is idempotent
          (in which case one of the operations isn't doing any remapping anyway).
          That is, applying the mapping twice is equivalent to applying it only once.
       
          A mapping is idempotent IFF:

              For a mapping from set A to set B, B is a superset of A and all items
              of B map to themselves.
              This is typically true of FlyEM supervoxels (A) and body IDs (B),
              since bodies always contain a supervoxel with matching ID.
    """
    def __init__(self, original_volume_service, labelmap_config):
        self.original_volume_service = original_volume_service # See VolumeService.service_chain
        validate(labelmap_config, LabelMapSchema, inject_defaults=True)

        # Convert relative path to absolute
        if not labelmap_config["file"].startswith('gs://') and not labelmap_config["file"].startswith("/"):
            abspath = os.path.abspath(labelmap_config["file"])
            labelmap_config["file"] = abspath
        
        self.labelmap_config = labelmap_config
        
        # These are computed on-demand and memoized for the sake of pickling support.
        # See __getstate__()
        self._mapper = None
        self._mapping_pairs = None
        self._compressed_mapping_pairs = None
        
        assert np.issubdtype(self.dtype, np.integer)
        
        self.apply_when_reading = labelmap_config["apply-when"] in ("reading", "reading-and-writing")
        self.apply_when_writing = labelmap_config["apply-when"] in ("writing", "reading-and-writing")

        self.missing_value_mode = labelmap_config["missing-value-mode"]

    def __getstate__(self):
        if self._compressed_mapping_pairs is None:
            # Load the labelmapping and then compress 
            mapping_pairs = np.asarray(self.mapping_pairs, order='C')
            self._compressed_mapping_pairs = (mapping_pairs.shape, mapping_pairs.dtype, lz4.frame.compress(mapping_pairs)) #@UndefinedVariable

        d = self.__dict__.copy()
        
        # Discard mapping pairs (will be reconstructed from compressed)
        d['_mapping_pairs'] = None
        
        # Discard mapper. It isn't pickleable
        d['_mapper'] = None
        return d

    @property
    def mapping_pairs(self):
        if self._mapping_pairs is None:
            if self._compressed_mapping_pairs is not None:
                shape, dtype, compressed = self._compressed_mapping_pairs
                self._mapping_pairs = np.frombuffer(lz4.frame.decompress(compressed), dtype).reshape(shape) #@UndefinedVariable
            else:
                self._mapping_pairs = load_labelmap(self.labelmap_config, '.')
                
                # Save RAM by converting to uint32 if possible (usually possible)
                if self._mapping_pairs.max() <= np.iinfo(np.uint32).max:
                    self._mapping_pairs = self._mapping_pairs.astype(np.uint32)
        return self._mapping_pairs

    @property
    def mapper(self):
        if not self._mapper:
            domain, codomain = self.mapping_pairs.transpose()
            self._mapper = LabelMapper(domain, codomain)
        return self._mapper

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
    def preferred_grid_offset(self):
        return self.original_volume_service.preferred_message_shape

    @property
    def bounding_box_zyx(self):
        return self.original_volume_service.bounding_box_zyx

    @property
    def available_scales(self):
        return self.original_volume_service.available_scales

    def get_subvolume(self, box_zyx, scale=0):
        volume = self.original_volume_service.get_subvolume(box_zyx, scale)
        assert np.issubdtype(volume.dtype, np.unsignedinteger), \
            f"LabelmappedVolumeService supports only unsigned integer types, not {volume.dtype}"

        if self.apply_when_reading:
            # TODO: Apparently LabelMapper can't handle non-contiguous arrays right now.
            #       (It yields incorrect results)
            #       Check to see if this is still a problem in the latest version of xtensor-python.
            volume = np.asarray(volume, order='C')

            if self.missing_value_mode == "identity":
                self.mapper.apply_inplace(volume, allow_unmapped=True)
            elif self.missing_value_mode == "error":
                self.mapper.apply_inplace(volume, allow_unmapped=False)
            elif self.missing_value_mode == "zero":
                volume = self.mapper.apply_with_default(volume)
            else:
                raise AssertionError("Unknown missing-value-mode")

        return volume

    def write_subvolume(self, subvolume, offset_zyx, scale=0):
        if self.apply_when_writing:
            # Copy first to avoid remapping user's input volume
            # (which they might want to reuse)
            subvolume = subvolume.copy(order='C')

            if self.missing_value_mode == "identity":
                self.mapper.apply_inplace(subvolume, allow_unmapped=True)
            elif self.missing_value_mode == "error":
                self.mapper.apply_inplace(subvolume, allow_unmapped=False)
            elif self.missing_value_mode == "zero":
                subvolume = self.mapper.apply_with_default(subvolume)
            else:
                raise AssertionError("Unknown missing-value-mode")

        self.original_volume_service.write_subvolume(subvolume, offset_zyx, scale)
