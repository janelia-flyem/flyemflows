from .base.workflow import Workflow
from .samplepoints import SamplePoints
from .copygrayscale import CopyGrayscale
from .decimatemeshes import DecimateMeshes
from .sparsemeshes import SparseMeshes
from .stitchedmeshes import StitchedMeshes

BUILTIN_WORKFLOWS = {
    'workflow': Workflow, # Base class, used for unit testing only
    'samplepoints': SamplePoints,
    'copygrayscale': CopyGrayscale,
    'decimatemeshes': DecimateMeshes,
    'sparsemeshes': SparseMeshes,
    'stitchedmeshes': StitchedMeshes
}

assert all([k == k.lower() for k in BUILTIN_WORKFLOWS.keys()]), \
    "Keys of BUILTIN_WORKFLOWS must be lowercase"

