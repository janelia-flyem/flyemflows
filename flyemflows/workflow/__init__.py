from .workflow import Workflow
from .samplepoints import SamplePoints

AVAILABLE_WORKFLOWS = {
    'samplepoints': SamplePoints
}
assert all([k == k.lower() for k in AVAILABLE_WORKFLOWS.keys()]), \
    "Keys of AVAILABLE_WORKFLOWS must be lowercase"

