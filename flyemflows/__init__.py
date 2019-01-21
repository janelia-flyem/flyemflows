import warnings
## Don't show the following warning from within pandas:
## FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
warnings.filterwarnings("ignore", module=r"pandas\..*", category=FutureWarning)

# TODO: This will dump faulthandler output with ordinary stdout,
#       which makes detecting possible segfaults in the worker
#       logs possibly time-consuming (depending on how much output
#       there is the worker logs.)
#       Consider writing faulthandler output to a separate file.
import faulthandler
faulthandler.enable()

# TODO:
# DVIDSparkServices had a lot of sophisticated configuration in its __init__ file.
# Some (or most) of it should be copied here.


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
