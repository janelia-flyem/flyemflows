import os
import time
import socket
import inspect
import logging
import psutil
from signal import SIGINT, SIGTERM, SIGKILL

logger = logging.getLogger(__name__)


class MemoryWatcher(object):
    def __init__(self, threshold_mb=1.0):
        self.hostname = socket.gethostname().split('.')[0]
        self.current_process = psutil.Process()
        self.initial_memory_usage = -1
        self.threshold_mb = threshold_mb
        self.ignore_threshold = False
    
    def __enter__(self):
        self.initial_memory_usage = self.current_process.memory_info().rss
        return self
    
    def __exit__(self, *args):
        pass

    def memory_increase(self):
        return self.current_process.memory_info().rss - self.initial_memory_usage
    
    def memory_increase_mb(self):
        return self.memory_increase() / 1024.0 / 1024.0

    def log_increase(self, logger, level=logging.DEBUG, note=""):
        if logger.isEnabledFor(level):
            caller_line = inspect.currentframe().f_back.f_lineno
            caller_file = os.path.basename( inspect.currentframe().f_back.f_code.co_filename )
            increase_mb = self.memory_increase_mb()
            
            if increase_mb > self.threshold_mb or self.ignore_threshold:
                # As soon as any message exceeds the threshold, show all messages from then on.
                self.ignore_threshold = True
                logger.log(level, "Memory increase: {:.1f} MB [{}] [{}:{}] ({})"
                                  .format(increase_mb, self.hostname, caller_file, caller_line, note) )


def is_process_running(pid, reap_zombie=True):
    """
    Return True if a process with the given PID
    is currently running, False otherwise.
    
    reap_zombie:
        If it's in zombie state, reap it and return False.
    """
    # Sending signal 0 to a pid will raise an OSError 
    # exception if the pid is not running, and do nothing otherwise.        
    # https://stackoverflow.com/a/568285/162094
    try:
        os.kill(pid, 0) # Signal 0
    except OSError:
        return False

    if not reap_zombie:
        return True

    proc = psutil.Process(pid)
    if proc.status() != psutil.STATUS_ZOMBIE:
        return True

    # Process is a zombie. Attempt to reap.
    try:
        os.waitpid(proc.pid, 0)
    except ChildProcessError:
        pass

    return False


def kill_if_running(pid, escalation_delay_seconds=10.0):
    """
    Kill the given process if it is still running.
    The process will be sent SIGINT, then SIGTERM if
    necessary (after escalation_delay seconds)
    and finally SIGKILL if it still hasn't died.
    
    This is similar to the behavior of the LSF 'bkill' command.
    
    Returns: True if the process was terminated 'nicely', or
             False it if it had to be killed with SIGKILL.
    """
    try:
        proc_cmd = ' '.join( psutil.Process(pid).cmdline() )
    except (psutil.NoSuchProcess, PermissionError, psutil.AccessDenied):
        return True

    _try_kill(pid, SIGINT)
    if not _is_still_running_after_delay(pid, escalation_delay_seconds):
        logger.info("Successfully interrupted process {}".format(pid))
        logger.info("Interrupted process was: " + proc_cmd)
        return True

    _try_kill(pid, SIGTERM)
    if not _is_still_running_after_delay(pid, escalation_delay_seconds):
        logger.info("Successfully terminated process {}".format(pid))
        logger.info("Terminated process was: " + proc_cmd)
        return True

    logger.warn("Process {} did not respond to SIGINT or SIGTERM.  Killing!".format(pid))
    
    # No more Mr. Nice Guy
    _try_kill(pid, SIGKILL, kill_children=True)
    logger.warn("Killed process was: " + proc_cmd)
    return False


def _try_kill(pid, sig, kill_children=False):
    """
    Attempt to terminate the process with the given ID via os.kill() with the given signal.
    If kill_children is True, then all child processes (and their children) 
    will be sent the signal as well, in unspecified order.
    """
    proc = psutil.Process(pid)
    procs_to_kill = [proc]
    
    if kill_children:
        for child in proc.children(recursive=True):
            procs_to_kill.append(child)
    
    for proc in procs_to_kill:
        try:
            os.kill(proc.pid, sig)
        except OSError as ex:
            if ex.errno != 3: # "No such process"
                raise


def _is_still_running_after_delay(pid, secs):
    still_running = is_process_running(pid)
    while still_running and secs > 0.0:
        time.sleep(2.0)
        secs -= 2.0
        still_running = is_process_running(pid)
    return still_running
    
