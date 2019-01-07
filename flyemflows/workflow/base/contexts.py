"""
Utilities for the Workflow base class.

The workflow needs to initialize and then tear
down various tools upon launch and exit.

Those initialization/tear-down processes are each encapuslated
as a different context manager defined in this file.

These are not meant to be used by callers other than
the Workflow base class itself.
"""
import os
from contextlib import contextmanager

@contextmanager
def environment_context(update_dict):
    """
    Context manager.
    Update the environment variables specified in the given dict
    when the context is entered, and restore the old environment when the context exits.
    
    Note:
        You can modify these or other environment variables while the context is active,
        those modifications will be lost when this context manager exits.
        (Your original environment is restored unconditionally.)
    """
    old_env = os.environ.copy()
    try:
        os.environ.update(update_dict)
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


