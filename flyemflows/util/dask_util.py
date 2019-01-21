import os
import getpass
import tempfile
import multiprocessing
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
    Persist and execute the given dask.Bag.
    The persisted Bag is returned.
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

class as_completed_synchronous:
    """
    For testing.
    A quick-n-dirty fake implementation of distributed.as_completed
    for use with the synchronous scheduler
    """
    def __init__(self):
        self.futures = []
    
    def add(self, f):
        self.futures.append(f)
    
    def __next__(self):
        if self.futures:
            return self.futures.pop(0)
        else:
            raise StopIteration()


class DebugClient:
    """
    Provides a small subset of the distributed.Client API, but behaves synchronously.
    Useful for convenient synchronous testing of code that is designed for the distributed cluster.
    """
    DEBUG = True

    def __init__(self, cluster_type='synchronous'):
        if cluster_type == "synchronous":
            self._ncores = 1
        else:
            self._ncores = multiprocessing.cpu_count()
    
    def ncores(self):
        return {'driver': self._ncores}
    
    def close(self):
        pass
    
    def scatter(self, data, *_):
        return FakeFuture(data)
    
    def compute(self, task):
        try:
            result = task.compute()
        except Exception as ex:
            return FakeFuture(None, task.key, ex)
        else:
            return FakeFuture(result, task.key)


class FakeFuture:
    """
    Future-like object, returned by DebugClient.compute() and DebugClient.scatter()
    """
    def __init__(self, result, key='fake', error=None):
        self._result = result
        self.key = key
        self._error = error
    
    def result(self):
        if self._error:
            raise self._error
        return self._result
    

