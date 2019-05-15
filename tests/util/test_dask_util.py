import pytest

import dask.bag as db
from dask import delayed

from flyemflows.util import drop_empty_partitions


def test_drop_empty_partitions():
    bag = db.from_delayed(map(delayed, [[1,2,3,4], [], [5,6,7], []]))
    assert bag.map_partitions(len).compute() == (4,0,3,0)
    assert drop_empty_partitions(bag).map_partitions(len).compute() == (4,3)

    empty_bag = db.from_delayed(map(delayed, [[], []]))
    assert empty_bag.map_partitions(len).compute() == (0,0)
    assert drop_empty_partitions(empty_bag).map_partitions(len).compute() == ()
    

if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.util.test_dask_util'])
