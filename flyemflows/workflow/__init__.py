from .workflow import Workflow
from .samplepoints import SamplePoints
from .copygrayscale import CopyGrayscale

AVAILABLE_WORKFLOWS = {
    'workflow': Workflow, # Base class, used for unit testing only
    'samplepoints': SamplePoints,
    'copygrayscale': CopyGrayscale
}

assert all([k == k.lower() for k in AVAILABLE_WORKFLOWS.keys()]), \
    "Keys of AVAILABLE_WORKFLOWS must be lowercase"

