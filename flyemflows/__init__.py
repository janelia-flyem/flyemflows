import sys
import signal

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

# Ensure SystemExit is raised if terminated via SIGTERM (e.g. by bkill).
signal.signal(signal.SIGTERM, lambda signum, stack_frame: sys.exit(0))

# Ensure SystemExit is raised if terminated via SIGUSR2.
# (The LSF cluster scheduler uses SIGUSR2 if the job's -W time limit has been exceeded.)
signal.signal(signal.SIGUSR2, lambda signum, stack_frame: sys.exit("Exiting due to SIGUSR2 (related: see manpage for 'bsub -W')"))

from . import _version
__version__ = _version.get_versions()['version']
