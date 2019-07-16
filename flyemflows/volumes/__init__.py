#
# The order these imports is rather important...
#
from .generic_schemas.geometry import BoundingBoxSchema, GeometrySchema

from .volume_service import VolumeService, VolumeServiceReader, VolumeServiceWriter

from .adapters import (TransposedVolumeService, NewAxisOrderSchema, ScaledVolumeService, RescaleLevelSchema,
                       LabelmappedVolumeService, LabelMapSchema, GrayscaleAdapters, SegmentationAdapters)

from .hdf5_volume_service import Hdf5VolumeService, Hdf5ServiceSchema, Hdf5VolumeSchema
from .boss_volume_service import BossVolumeServiceReader, BossServiceSchema
from .brainmaps_volume_service import BrainMapsVolumeServiceReader, BrainMapsServiceSchema
from .dvid_volume_service import DvidVolumeService, DvidGrayscaleServiceSchema, DvidSegmentationServiceSchema, DvidSegmentationVolumeSchema
from .n5_volume_service import N5VolumeService, N5ServiceSchema
from .zarr_volume_service import ZarrVolumeService, ZarrServiceSchema, ZarrVolumeSchema
from .slice_files_volume_service import SliceFilesVolumeService, SliceFilesServiceSchema, SliceFilesVolumeSchema

from .generic_schemas.volumes import (GrayscaleVolumeSchema, SegmentationVolumeSchema, SegmentationVolumeListSchema)
