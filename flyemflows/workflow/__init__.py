from .base.workflow import Workflow
from .chunkedbodymeshes import ChunkedBodyMeshes
from .connectedcomponents import ConnectedComponents
from .contingencytable import ContingencyTable
from .contingentrelabel import ContingentRelabel
from .copygrayscale import CopyGrayscale
from .createmeshes import CreateMeshes
from .copysegmentation import CopySegmentation
from .decimatemeshes import DecimateMeshes
from .findadjacencies import FindAdjacencies
from .gridmeshes import GridMeshes
from .labelmapcopy import LabelmapCopy
from .maskedcopy import MaskedCopy
from .masksegmentation import MaskSegmentation
from .mitodistances import MitoDistances
from .mitorepair import MitoRepair
from .mitostats import MitoStats
from .roistats import RoiStats
from .samplepoints import SamplePoints
from .sparseblockstats import SparseBlockstats
from .sparsemeshes import SparseMeshes
from .stitchedmeshes import StitchedMeshes
from .svdecimate import SVDecimate


BUILTIN_WORKFLOWS = [
    Workflow,  # Base class, used for unit testing only
    ChunkedBodyMeshes,
    ConnectedComponents,
    ContingencyTable,
    ContingentRelabel,
    CopyGrayscale,
    CopySegmentation,
    CreateMeshes,
    DecimateMeshes,
    FindAdjacencies,
    GridMeshes,
    LabelmapCopy,
    MaskedCopy,
    MaskSegmentation,
    MitoDistances,
    MitoRepair,
    MitoStats,
    RoiStats,
    SamplePoints,
    SparseBlockstats,
    SparseMeshes,
    StitchedMeshes,
    SVDecimate,
]
