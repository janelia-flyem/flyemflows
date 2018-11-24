from collections import defaultdict
from dask.bag import Bag
from neuclease.util import Timer

def persist_and_execute(bag, description, logger=None):
    """
    Persist and execute the given RDD or iterable.
    The persisted RDD is returned (in the case of an iterable, it may not be the original)
    """
    assert isinstance(bag, Bag)
    if logger:
        logger.info(f"{description}...")

    with Timer() as timer:
        bag.persist()
        count = bag.count() # force eval
        parts = bag.npartitions()
        partition_counts = bag.map_partitions(lambda part: [sum(1 for _ in part)]).collect()
        histogram = defaultdict(lambda : 0)
        for c in partition_counts:
            histogram[c] += 1
        histogram = dict(histogram)

    if logger:
        logger.info(f"{description} (N={count}, P={parts}, P_hist={histogram}) took {timer.timedelta}")
    
    return bag
