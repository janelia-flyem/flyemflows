from .base.workflow import Workflow
from .copygrayscale import CopyGrayscale
from .decimatemeshes import DecimateMeshes
from .findadjacencies import FindAdjacencies
from .samplepoints import SamplePoints
from .sparsemeshes import SparseMeshes
from .stitchedmeshes import StitchedMeshes

BUILTIN_WORKFLOWS = [
    Workflow, # Base class, used for unit testing only
    CopyGrayscale,
    DecimateMeshes,
    FindAdjacencies,
    SamplePoints,
    SparseMeshes,
    StitchedMeshes,
]
