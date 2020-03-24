from .base.workflow import Workflow
from .connectedcomponents import ConnectedComponents
from .contingencytable import ContingencyTable
from .contingentrelabel import ContingentRelabel
from .copygrayscale import CopyGrayscale
from .createmeshes import CreateMeshes
from .copysegmentation import CopySegmentation
from .decimatemeshes import DecimateMeshes
from .findadjacencies import FindAdjacencies
from .labelmapcopy import LabelmapCopy
from .masksegmentation import MaskSegmentation
from .roistats import RoiStats
from .samplepoints import SamplePoints
from .sparseblockstats import SparseBlockstats
from .sparsemeshes import SparseMeshes
from .stitchedmeshes import StitchedMeshes


BUILTIN_WORKFLOWS = [
    Workflow, # Base class, used for unit testing only
    ConnectedComponents,
    ContingencyTable,
    ContingentRelabel,
    CopyGrayscale,
    CopySegmentation,
    DecimateMeshes,
    FindAdjacencies,
    LabelmapCopy,
    MaskSegmentation,
    RoiStats,
    SamplePoints,
    SparseBlockstats,
    SparseMeshes,
    StitchedMeshes,
    CreateMeshes
]
