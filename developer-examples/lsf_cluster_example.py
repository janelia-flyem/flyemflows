"""
Example code to launch a dask cluster, when you aren't using flyemflows.

This example is not related to flyemflows per se.
It's just a simple example for new dask users, showing how to spin
up a dask distributed cluster on LSF, using LSFCluster() from dask_jobqueue.

(Normally when you run a Workflow using flyemflows,
the cluster is automatically started for you, so there is
no need to use this code in that case.)

For an example dask configuration to use with LSFCluster,
see example-dask-config.yaml

To run this example, start ipython like this:
    $ export DASK_CONFIG=example-dask-config.yaml
    $ ipython

And then paste this code into the terminal.
"""

import time
import getpass

import dask
import dask.bag as db
from distributed import Client

def init_cluster(num_workers, wait_for_all_workers=True):
    """
    Start up a dask cluster, optionally wait until all workers have been launched,
    and then return the resulting distributed.Client object.
    
    Args:
        num_workers:
            How many workers to launch.
        wait_for_all_workers:
            If True, pause until all workers have been launched before returning.
            Otherwise, just wait for a single worker to launch.
    
    Returns:
        distributed.Client
    """
    # Local import: LSFCluster probably isn't importable on your local machine,
    # so it's nice to avoid importing it when you're just running local tests without a cluster.
    from dask_jobqueue import LSFCluster
    cluster = LSFCluster(ip='0.0.0.0')
    cluster.scale(num_workers)
    
    required_workers = 1
    if wait_for_all_workers:
        required_workers = num_workers
        
    client = Client(cluster)
    while (wait_for_all_workers and
           client.status == "running" and
           len(cluster.scheduler.workers) < required_workers):
        print(f"Waiting for {required_workers - len(cluster.scheduler.workers)} workers...")
        time.sleep(1.0)
    
    return client

def main():
    USER = getpass.getuser()
    
    # See example-dask-config.yaml for explanations.
    dask.config.set({'jobqueue':
                        {'lsf':
                          {'cores': 1,
                           'memory': '15GB',
                           'walltime': '01:00',
                           'log-directory': 'dask-logs',
                           'local-directory': f'/scratch/{USER}',
                           'use-stdin': True    # Implementation detail regarding how bsub is called by dask-jobqueue.
                                                # Under Janelia's LSF configuration, this must be set to 'True'.

                    }}})
    
    client = init_cluster(2)
    
    try:
        def double(x):
            return 2*x
    
        bag = db.from_sequence(range(100))
        doubled = bag.map(double).compute()
        print(doubled)
    
    finally:
        client.close()
        client.cluster.close()


if __name__ == "__main__":
    main()
