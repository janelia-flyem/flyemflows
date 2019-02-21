import pytest
from flyemflows.util import auto_retry

# Reset this in case other tests have disabled auto_retry
import flyemflows.util._auto_retry #@UnusedImport
flyemflows.util._auto_retry.FLYEMFLOWS_DISABLE_AUTO_RETRY = False

COUNTER = None

@pytest.fixture
def setup_counter():
    global COUNTER
    COUNTER = 3

def _check_counter():
    global COUNTER
    COUNTER -= 1
    assert COUNTER == 0, f'counter is {COUNTER}'

def test_failed_retry(setup_counter):
    
    @auto_retry(2, 0.0)
    def should_fail():
        _check_counter()
    
    try:
        should_fail()
    except AssertionError:
        pass
    else:
        assert False, "should_fail() didn't fail!"

def test_successful_retry(setup_counter):        

    @auto_retry(3, 0.0)
    def should_succeed():
        _check_counter()

    try:
        should_succeed()
    except AssertionError:
        assert False, "should_succeed() didn't succeed!"

def test_failed_with_predicate(setup_counter):
    def predicate(ex):
        return '1' in ex.args[0]

    @auto_retry(2, 0.0, predicate=predicate)
    def should_fail():
        _check_counter()
    
    try:
        should_fail()
    except AssertionError:
        pass
    else:
        assert False, "should_fail() didn't fail!"

def test_success_with_predicate(setup_counter):
    def predicate(ex):
        return '1' in ex.args[0] or '2' in ex.args[0]
    
    @auto_retry(3, 0.0, predicate=predicate )
    def should_succeed():
        _check_counter()
    
    try:
        should_succeed()
    except AssertionError:
        assert False, "should_succeed() didn't succeed!"


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.util.test_auto_retry'])
