import os
import copy
import socket
import logging
import getpass
import tempfile
import multiprocessing
from collections import defaultdict

import dask
import dask.bag
import dask.config
import dask.dataframe
from dask.bag import Bag

import distributed.config
from distributed.utils import parse_bytes
from distributed.client import futures_of

from neuclease.util import Timer

from .dask_schema import DaskConfigSchema
from confiddler import load_config, dump_config, validate

from . import extract_ip_from_link

logger = logging.getLogger(__name__)

def update_jobqueue_config_with_defaults(cluster_type):
    """
    Read the jobqueue.lsf configuration and fill in
    missing values for ncpus, mem, and local-directory.
    Also, set tempfile.tempdir according to the 'local-directory' setting.
    """
    # Make sure we've at least got the defaults.
    import dask_jobqueue
    builtin_defaults = load_config(os.path.dirname(dask_jobqueue.__file__) + '/jobqueue.yaml')
    cfg = dask.config.update(dask.config.config, builtin_defaults, priority='old')
    dask.config.set(cfg)

    # Must import LSFCluster, SGECluster, etc. first, or else the
    # dask.config data for lsf isn't fully populated yet.
    if cluster_type == "lsf":
        _update_lsf_settings()
    if cluster_type == "sge":
        _update_sge_settings()
    if cluster_type == "slurm":
        _update_slurm_settings()

    _set_local_directory(cluster_type)

def _update_lsf_settings():
    from dask_jobqueue import LSFCluster #@UnresolvedImport @UnusedImport
    # 'ncpus' is how many CPUs are RESERVED for the LSF job.
    # By default, set it to the number of CPUs the workers will actually use ('cores')
    ncpus = dask.config.get("jobqueue.lsf.ncpus", -1)
    if not ncpus or ncpus == -1:
        ncpus = dask.config.get("jobqueue.lsf.cores")
        dask.config.set({"jobqueue.lsf.ncpus": ncpus})

    # Similar to above, the difference between 'mem' and 'memory' is that the former
    # specifies the memory to reserve in LSF, whereas the latter is actually used
    # by Dask workers to determine when they've exceeded their limits.
    memory = dask.config.get("jobqueue.lsf.memory", None)
    mem = dask.config.get("jobqueue.lsf.mem", 0)
    if memory and not mem:
        mem = parse_bytes(memory)
        dask.config.set({"jobqueue.lsf.mem": mem})

def _update_sge_settings():
    # No settings to change (for now)
    # job-cpu and job-mem are given suitable defaults if they aren't specified.
    from dask_jobqueue import SGECluster #@UnresolvedImport @UnusedImport

def _update_slurm_settings():
    # No settings to change (for now)
    from dask_jobqueue import SlurmCluster #@UnresolvedImport @UnusedImport


def _set_local_directory(cluster_type):
    # This specifies where dask workers will dump cached data
    local_dir = dask.config.get(f"jobqueue.{cluster_type}.local-directory", None)
    if local_dir:
        return

    user = getpass.getuser()
    local_dir = None
    for d in [f"/scratch/{user}", f"/tmp/{user}"]:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            continue
        else:
            local_dir = d
            dask.config.set({f"jobqueue.{cluster_type}.local-directory": local_dir})

            # Set tempdir, too.
            tempfile.tempdir = local_dir
            
            # Forked processes will use this for tempfile.tempdir
            os.environ['TMPDIR'] = local_dir
            break
    
    if local_dir is None:
        raise RuntimeError("Could not create a local-directory in any of the standard places.")


def dump_dask_config(path):
    """
    Dump the current dask.config.config to the given path.
    
    Note: If jobqueue settings are present in the config,
          only the lsf settings are included.
          Others are omitted. 
    """
    config = copy.deepcopy(dask.config.config)
    if dask.config.get('jobqueue.lsf', None):
        # Delete all the other jobqueue settings we
        # don't care about (i.e. other cluster types)
        lsf = config['jobqueue']['lsf']
        del config['jobqueue']
        config['jobqueue'] = { 'lsf': lsf }
    dump_config(config, path)


def load_and_overwrite_dask_config(cluster_type, dask_config_path=None, overwrite=False):
    """
    Load dask config, inject defaults for (selected) missing entries,
    and optionally overwrite in-place.

    Note: Also re-initializes the distributed logging configuration.
    """
    if dask_config_path is None and 'DASK_CONFIG' in os.environ:
        dask_config_path = os.environ["DASK_CONFIG"]
    dask_config_path = dask_config_path or 'dask-config.yaml'

    dask_config_path = os.path.abspath(dask_config_path)
    if os.path.exists(dask_config_path):
        # Check for completely empty dask config file
        from ruamel.yaml import YAML
        yaml = YAML()
        config = yaml.load(open(dask_config_path, 'r'))
        if not config:
            dask_config = {}
            validate(dask_config, DaskConfigSchema, inject_defaults=True)
        else:
            dask_config = load_config(dask_config_path, DaskConfigSchema)
    else:
        dask_config = {}
        validate(dask_config, DaskConfigSchema, inject_defaults=True)

    # Don't pollute the config file with extra jobqueue parameters we aren't using
    if "jobqueue" in dask_config:
        for key in list(dask_config["jobqueue"].keys()):
            if key != cluster_type:
                del dask_config["jobqueue"][key]

        if len(dask_config["jobqueue"]) == 0:
            del dask_config["jobqueue"]

    if overwrite:
        dump_config(dask_config, dask_config_path)

    # This environment variable is recognized by dask itself
    os.environ["DASK_CONFIG"] = dask_config_path
    dask.config.paths.append(dask_config_path)
    dask.config.refresh()

    # Must be imported this way due to aliased name 'config' in distributed.__init__
    from distributed.config import initialize_logging
    initialize_logging(dask.config.config)


def run_on_each_worker(func, client=None, once_per_machine=False, return_hostnames=True):
    """
    Run the given function once per worker (or once per worker machine).
    Results are returned in a dict of { worker: result }
    
    Args:
        func:
            Must be picklable.
        
        client:
            If provided, must be a distributed Client object.
            If None, it is assumed you are not using a distributed cluster,
            and so the function will only be run once, on the driver (synchronously).
        
        once_per_machine:
            Ensure that the function is only run once per machine,
            even if your cluster is configured to run more than one
            worker on each node.
        
        return_hostnames:
            If True, result keys use hostnames instead of IPs.
    Returns:
        dict:
        { 'ip:port' : result } OR
        { 'hostname:port' : result }
    """
    try:
        funcname = func.__name__
    except AttributeError:
        funcname = 'unknown function'

    if client is None:
        # Assume non-distributed scheduler (either synchronous or processes)
        if return_hostnames:
            results = {f'tcp://{socket.gethostname()}': func()}
        else:
            results = {'tcp://127.0.0.1': func()}
        logger.info(f"Ran {funcname} on the driver only")
        return results

    all_worker_hostnames = client.run(socket.gethostname)
    if not once_per_machine:
        worker_hostnames = all_worker_hostnames

    if once_per_machine:
        machines = set()
        worker_hostnames = {}
        for address, name in all_worker_hostnames.items():
            ip = address.split('://')[1].split(':')[0]
            if ip not in machines:
                machines.add(ip)
                worker_hostnames[address] = name
    
    workers = list(worker_hostnames.keys())
    with Timer(f"Running {funcname} on {len(workers)} workers", logger):
        results = client.run(func, workers=workers)
    
    if not return_hostnames:
        return results

    final_results = {}
    for address, result in results.items():
        hostname = worker_hostnames[address]
        ip = extract_ip_from_link(address)
        final_results[address.replace(ip, hostname)] = result

    return final_results


def persist_and_execute(bag, description=None, logger=None, optimize_graph=True, return_counts=False):
    """
    Persist and execute the given dask.Bag.
    The persisted Bag is returned.
    """
    assert isinstance(bag, Bag)
    if logger and description:
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

    if logger and description:
        logger.info(f"{description} (N={count}, P={parts}, P_hist={histogram}) took {timer.timedelta}")
    
    if return_counts:
        return bag, count, parts, histogram
    else:
        return bag


def drop_empty_partitions(bag_or_df):
    if isinstance(bag_or_df, dask.bag.Bag):
        return drop_empty_bag_partitions(bag_or_df)
    if isinstance(bag_or_df, dask.dataframe.DataFrame):
        return drop_empty_ddf_partitions(bag_or_df)
    raise AssertionError(f"Unsupported input type: {type(bag_or_df)}")


def drop_empty_bag_partitions(bag):
    """
    When bags are created by filtering or grouping from a different bag,
    it retains the original bag's partition count, even if a lot of the
    partitions become empty.
    Those extra partitions add overhead, so it's nice to discard them.
    This function drops the empty partitions.
    Inspired by: https://stackoverflow.com/questions/47812785/remove-empty-partitions-in-dask
    """
    bag = bag.persist()
    def get_len(partition):
        # If the bag is the result of bag.filter(),
        # then each partition is actually a 'filter' object,
        # which has no __len__.
        # In that case, we must convert it to a list first.
        if hasattr(partition, '__len__'):
            return len(partition)
        return len(list(partition))
    partition_lengths = bag.map_partitions(get_len).compute()
    
    # Convert bag partitions into a list of 'delayed' objects
    lengths_and_partitions = zip(partition_lengths, bag.to_delayed())
    
    # Drop the ones with empty partitions
    partitions = (p for l,p in lengths_and_partitions if l > 0)
    
    # Convert from list of delayed objects back into a Bag.
    return dask.bag.from_delayed(partitions)
    

def drop_empty_ddf_partitions(df):
    """
    https://stackoverflow.com/questions/47812785/remove-empty-partitions-in-dask
    """
    df = df.persist()
    ll = list(df.map_partitions(len).compute())
    df_delayed = df.to_delayed()
    df_delayed_new = list()
    pempty = None
    for ix, n in enumerate(ll):
        if 0 == n:
            pempty = df.get_partition(ix)
        else:
            df_delayed_new.append(df_delayed[ix])
    if pempty is not None:
        df = dask.dataframe.from_delayed(df_delayed_new, meta=pempty)
    return df


def release_collection(collection, client=None):
    """
    An explicit unpersist() function for dask collections,
    for when you can't merely release the reference because
    it is held by a downstream persisted() task.

    Copied from:
        - https://github.com/dask/dask/issues/2492
        - https://stackoverflow.com/questions/44797668
    """
    if client is None or isinstance(client, DebugClient):
        return

    for future in futures_of(collection):
        future.release()


class as_completed_synchronous:
    """
    For testing.
    A quick-n-dirty fake implementation of distributed.as_completed
    for use with the synchronous scheduler.
    """
    def __init__(self, futures=None, loop=None, with_results=False, raise_errors=True):
        self.futures = futures or []
        self.with_results = with_results
        self.raise_errors = raise_errors

    def add(self, f):
        self.futures.append(f)

    # Note: distributed.as_completed does not support __len__

    def __iter__(self):
        return self

    def __next__(self):
        if not self.futures:
            raise StopIteration

        f = self.futures.pop(0)
        if not self.with_results:
            return f

        try:
            return (f, f.result())
        except:
            if not self.raise_errors:
                return (f, None)
            raise


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
        self.cluster = None
    
    def ncores(self):
        return {'driver': self._ncores}
    
    def close(self):
        pass
    
    def scatter(self, data, *_):
        return FakeFuture(data)
    
    def map(self, func, *iterables, key=None, workers=None, retries=None, resources=None,
            priority=0, allow_other_workers=False, fifo_timeout="100 ms", actor=False,
            actors=False, pure=None, batch_size=None,
            **kwargs):
        """
        Synchronous implementation of Client.map()
        https://docs.dask.org/en/latest/futures.html#distributed.Client.map

        In this version, all keyword arguments are ignored except 'key' and 'kwargs'.
        """
        futures = []
        for args in zip(*iterables):
            try:
                result = func(*args, **kwargs)
            except Exception as ex:
                f = FakeFuture(None, key, ex)
            else:
                f = FakeFuture(result, key)
            futures.append(f)
        return futures

    def compute(self, task):
        try:
            result = task.compute()
        except Exception as ex:
            return FakeFuture(None, task.key, ex)
        else:
            return FakeFuture(result, task.key)

    def rebalance(self, *args, **kwargs):
        pass


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
