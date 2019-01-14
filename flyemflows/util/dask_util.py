import os
import getpass
import tempfile
from collections import defaultdict

import dask
from dask.bag import Bag
from distributed.utils import parse_bytes

from neuclease.util import Timer

def update_lsf_config_with_defaults():
    """
    Read the jobqueue.lsf configuration and fill in
    missing values for ncpus, mem, and local-directory.
    Also, set tempfile.tempdir according to the 'local-directory' setting.
    """
    # Must import LSFCluster first, or else the
    # dask.config data for lsf isn't fully populated yet.
    from dask_jobqueue import LSFCluster #@UnresolvedImport

    # 'ncpus' is how many CPUs are RESERVED for the LSF job.
    # By default, set it to the number of CPUs the workers will actually use ('cores')
    ncpus = dask.config.get("jobqueue.lsf.ncpus", -1)
    if ncpus == -1:
        ncpus = dask.config.get("jobqueue.lsf.cores")
        dask.config.set({"jobqueue.lsf.ncpus": ncpus})

    # Similar to above, the difference between 'mem' and 'memory' is that the former
    # specifies the memory to reserve in LSF, whereas the latter is actually used
    # by Dask workers to determine when they've exceeded their limits.
    mem = dask.config.get("jobqueue.lsf.mem", None)
    if not mem:
        memory = dask.config.get("jobqueue.lsf.memory", None)
        if memory:
            mem = parse_bytes(memory)
            dask.config.set({"jobqueue.lsf.mem": mem})

    # This specifies where dask workers will dump cached data
    local_dir = dask.config.get("jobqueue.lsf.local-directory", None)
    if not local_dir:
        user = getpass.getuser()
        local_dir = f"/scratch/{user}"
        dask.config.set({"jobqueue.lsf.local-directory": local_dir})
        
    # Set tempdir, too.
    tempfile.tempdir = local_dir
    
    # Forked processes will use this for tempfile.tempdir
    os.environ['TMPDIR'] = local_dir

def persist_and_execute(bag, description, logger=None, optimize_graph=True):
    """
    Persist and execute the given RDD or iterable.
    The persisted RDD is returned (in the case of an iterable, it may not be the original)
    """
    assert isinstance(bag, Bag)
    if logger:
        logger.info(f"{description}...")

    with Timer() as timer:
        bag = bag.persist(optimize_graph=optimize_graph)
        count = bag.count().compute() # force eval
        parts = bag.npartitions
        partition_counts = bag.map_partitions(lambda part: [sum(1 for _ in part)]).compute()
        histogram = defaultdict(lambda : 0)
        for c in partition_counts:
            histogram[c] += 1
        histogram = dict(histogram)

    if logger:
        logger.info(f"{description} (N={count}, P={parts}, P_hist={histogram}) took {timer.timedelta}")
    
    return bag
