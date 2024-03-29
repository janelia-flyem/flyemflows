import functools
import logging

# The following global flag is used for testing,
# when @auto_retry obscures/delays legitimate failures.
# 
# This module is named '_auto_retry' instead of 'auto_retry'
# to make this variable accessible via:
# 
#   import flyemflows.util._autoretry.FLYEMFLOWS_DISABLE_AUTO_RETRY
#   flyemflows.util._autoretry.FLYEMFLOWS_DISABLE_AUTO_RETRY = True
#
FLYEMFLOWS_DISABLE_AUTO_RETRY = False

def auto_retry(total_tries=2, pause_between_tries=10.0, logging_name=None, predicate=None):
    """
    Returns a decorator.
    If the decorated function fails for any reason,
    pause for a bit and then retry until it has been called total_tries times.
    
    total_tries:
        How many times to execute the function in total.
        (Example: If total_tries = 1, the function is not retried at all.)
    
    pause_between_tries:
        How long to wait before retrying the function.
    
    logging_name:
        If any retries are necessary, a warning will be logged
        to the logger with the given name.
    
    predicate:
        (Optional)
        Should be a callable with signature: f(exception) -> bool
        It will be called once per retry:
          - If it returns true, we continue retrying as usual until total_tries
            have been exhausted.
          - If it returns False, the retries are aborted, regardless of total_tries.
    """
    assert total_tries >= 1
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            remaining_tries = total_tries
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as ex:
                    if FLYEMFLOWS_DISABLE_AUTO_RETRY or (predicate is not None and not predicate(ex)):
                        raise
                    remaining_tries -= 1
                    if remaining_tries == 0:
                        raise
                    if logging_name:
                        logger = logging.getLogger(logging_name)
                        logger.warning("Call to '{}' failed with error: {}.".format(func.__name__, repr(ex)))
                        logger.warning("Retrying {} more times".format( remaining_tries ))
                    import time
                    time.sleep(pause_between_tries)
        wrapper.__wrapped__ = func # Emulate python 3 behavior of @functools.wraps
        return wrapper
    return decorator
