import os
import re
import sys
import time
import ctypes
import socket
import smtplib
import getpass
import logging
import datetime
import traceback
from collections.abc import Collection
from email.mime.text import MIMEText
from ctypes.util import find_library
from contextlib import contextmanager
from subprocess import Popen, PIPE, TimeoutExpired, run as subprocess_run

import vigra
import psutil
import numpy as np
import scipy.ndimage
from skimage.util import view_as_blocks
from skimage.measure import block_reduce
from skimage.transform import downscale_local_mean

from dvidutils import downsample_labels
from neuclease.util import Timer, parse_timestamp, downsample_labels_3d_suppress_zero, bincount_last_axis

logger = logging.getLogger(__name__)

try:
    libc = ctypes.cdll.msvcrt # Windows
except (OSError, AttributeError):
    libc = ctypes.cdll.LoadLibrary(find_library('c'))

def replace_default_entries(array, default_array, marker=-1):
    """
    Overwrite all entries in array that match the given
    marker with the corresponding entry in default_array.
    """
    new_array = np.array(array)
    default_array = np.asarray(default_array)
    assert new_array.shape == default_array.shape
    new_array[:] = np.where(new_array == marker, default_array, new_array)
    
    if isinstance(array, np.ndarray):
        array[:] = new_array
    elif isinstance(array, list):
        # Slicewise assignment is broken for Ruamel sequences,
        # which are often passed to this function.
        # array[:] = new_array.list() # <-- broken
        # https://bitbucket.org/ruamel/yaml/issues/176/commentedseq-does-not-support-slice
        #
        # Use one-by-one item assignment instead:
        for i,val in enumerate(new_array.tolist()):
            array[i] = val
    else:
        raise RuntimeError("This function supports arrays and lists, nothing else.")

DOWNSAMPLE_METHODS = ['subsample', 'zoom', 'grayscale', 'block-mean', 'mode', 'labels', 'label', 'labels-numba']
def downsample(volume, factor, method):
    if method is None:
        # We assume these are labels
        if volume.dtype in (np.uint64, np.uint32):
            method = 'subsample'
        else:
            method = 'block-mean'

    assert method in DOWNSAMPLE_METHODS
    assert (np.array(volume.shape) % factor == 0).all(), \
        "Volume dimensions must be a multiple of the downsample factor."
    
    if not isinstance(factor, Collection):
        factor = (factor,)*volume.ndim

    if method == 'subsample':
        strided_slicing = tuple(slice(None, None, f) for f in factor)
        return volume[strided_slicing].copy('C')
    
    if method == 'block-mean':
        return block_downsample(volume, factor)

    if method in ('zoom', 'grayscale'):
        # vigra is 2.7x faster than scipy, but it complains for small images:
        # 
        #  Precondition violation!
        #  resizeImage(): Each output axis must have length > 1.)
        #
        # Furthermore, it seems to be somewhat unstable even for small
        # images that satisfy the above condition.
        # Therefore, use vigra for non-tiny images,
        # and use scipy for the tiny stuff.
        newshape = np.array(volume.shape) // factor
        if (newshape >= 10).all():
            return vigra_sampling_resize(volume, newshape)
        else:
            return scipy.ndimage.zoom(volume, 1/factor)

    if method == 'mode':
        if len(set(factor)) > 1:
            raise RuntimeError("The 'mode' downsampling implementation doesn't support anisotric downsampling.")
        return downsample_labels(volume, factor, False)
    if method in ('labels', 'label'):  # synonyms
        if len(set(factor)) > 1:
            raise RuntimeError("The 'labels' downsampling implementation doesn't support anisotric downsampling.")
        return downsample_labels(volume, factor, True)
    if method == 'labels-numba':
        reduced_output, _reduced_box = downsample_labels_3d_suppress_zero(volume, factor)
        return reduced_output
    if method == 'labels-numpy-lowval':
        return block_mode_via_bincounts(volume, factor, True)

    raise AssertionError("Shouldn't get here.")


def block_mode_via_bincounts(volume, factor, suppress_zero=False):
    """
    Same as dvidutils.downsample_labels(), but implemented using only numpy.
    However, this relies on bincount, and therefore cannot handle inputs with
    arbitrarily high label values.

    And it's not as fast anyway.
    """
    assert (np.array(volume.shape) % factor == 0).all(), \
        "Volume dimensions must be a multiple of the downsample factor."

    if not isinstance(factor, Collection):
        factor = (factor,)*volume.ndim

    view = view_as_blocks(volume, factor)
    view = view.reshape((*view.shape[:3], np.prod(factor)))
    bc = bincount_last_axis(view)

    if suppress_zero:
        return np.argmax(bc[..., 1:], axis=-1) + 1

    return np.argmax(bc, axis=-1)


def block_downsample(volume, factor):
    """
    Simple downsampling by averaging pixels that fall under each output voxel.
    """
    dtype = volume.dtype
    assert (np.array(volume.shape) % factor == 0).all(), \
        "Volume dimensions must be a multiple of the downsample factor."

    if not isinstance(factor, Collection):
        factor = (factor,)*volume.ndim

    # Special case for uint8 grayscale downsampling;
    # Use uint16 intermediate value instead of instead
    # of float (as long as the sum of the bin voxels always fits in a uint16).
    if volume.dtype == np.uint8 and np.prod(factor) <= 256:
        sums = block_reduce(volume, factor, lambda a, axis: a.sum(axis, np.uint16))
        denominator = np.prod(factor)
        return (sums // denominator).astype(np.uint8)
    elif np.issubdtype(volume.dtype, np.integer):
        # numpy/scipy will convert integers to float64
        # unless we pre-convert to lower precision first.
        volume = volume.astype(np.float32, order='C')
    
    return downscale_local_mean(volume, factor).astype(dtype, copy=False)


def _block_downsample_2(volume, factor):
    """
    Do not use.

    Alternative to block_downsample() that attempts to save time by
    summing bins along each axis in turn, rather than summing across
    entire blocks right away.

    It turns out this is not usually as fast, probably due to the RAM
    allocations needed for the intermediate results.  There are some inputs
    for which this is better (especially high-dimensional volumes),
    but alas, not for the common 3D, 2x downsampling case.
    """
    assert (np.array(volume.shape) % factor == 0).all(), \
        "Volume dimensions must be a multiple of the downsample factor."

    if not isinstance(factor, Collection):
        factor = (factor,)*volume.ndim

    dtype = volume.dtype
    if volume.dtype == np.uint8 and np.prod(factor) <= 256:
        volume = volume.astype(np.uint16)
    elif np.issubdtype(volume.dtype, np.integer):
        # Avoid overflow issues at the expense of precision
        volume = volume.astype(np.float32, order='C')

    for f in reversed(factor):
        volume = np.rollaxis(volume, -1, 0)
        binned_shape = (volume.shape[0] // f, f, *volume.shape[1:])
        volume = volume.reshape(binned_shape).sum(1, volume.dtype)

    if np.issubdtype(volume.dtype, np.integer):
        volume //= np.prod(factor)
    else:
        volume /= np.prod(factor)

    return volume.astype(dtype, copy=False)


def vigra_sampling_resize(volume, newshape):
    newshape = tuple(newshape)
    volume_f = np.asarray(volume, dtype=np.float32)
    
    axes = 'zyx'[3-volume_f.ndim:]
    volume_f = vigra.taggedView(volume_f, axes)
    
    resized = vigra.sampling.resize(volume_f, newshape)
    return np.asarray(resized, dtype=volume.dtype)


def upsample(volume, factor):
    """
    Upsample the volume into a larger volume by duplicating values.
    """
    ndim = volume.ndim
    if isinstance(factor, Collection):
        factor = tuple(factor)
    else:
        factor = (factor,) * ndim

    upsampled_data = np.empty( np.array(volume.shape) * factor, dtype=volume.dtype )
    v = view_as_blocks(upsampled_data, factor)

    slicing = ((slice(None),) * ndim) + ((None,) * ndim)
    v[:] = volume[slicing]
    return upsampled_data

def get_localhost_ip_address():
    """
    Return this machine's own IP address, as seen from the network
    (e.g. 192.168.1.152, not 127.0.0.1)
    """
    try:
        # Determine our own machine's IP address
        # This method is a little hacky because it requires
        # making a connection to some arbitrary external site,
        # but it seems to be more reliable than the method below. 
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("google.com",80))
        ip_addr = s.getsockname()[0]
        s.close()
        
    except socket.gaierror:
        # Warning: This method is simpler, but unreliable on some networks.
        #          For example, on a home Verizon FiOS network it will error out in the best case,
        #          or return the wrong IP in the worst case (if you haven't disabled their DNS
        #          hijacking on your router)
        ip_addr = socket.gethostbyname(socket.gethostname())
    
    return ip_addr
    

def is_port_open(port):
    """
    Return True if the given port is already open on the local machine.
    
    https://stackoverflow.com/questions/19196105/python-how-to-check-if-a-network-port-is-open-on-linux
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect_ex(('127.0.0.1',port))
    except ConnectionRefusedError:
        return False
    else:
        return True
    finally:
        sock.close()


def extract_ip_from_link(link):
    """
    Given a link with an IP address instead of a hostname,
    returns the IP (as a string).
    If the link does not contain an IP, returns None.

    Example inputs:
    
        http://127.0.0.1/foo
        http://10.36.111.11:38817/status
        tcp://10.36.111.11:38003
        
    """
    m = re.match(r'.*://(\d+\.\d+\.\d+.\d+).*', link)
    if m:
        return m.groups()[0]
    else:
        return None


def find_processes(search_string):
    """
    Search all processes on the machine for one whose command
    line invocation contains the given search string.
    
    Returns a list of psutil.Process objects of matching processes.
    The list will be empty if no matches were found.

    Note: Certain root-level processes cannot be scanned by this function.
    """
    procs = []
    for p in psutil.process_iter():
        try:
            if search_string in ' '.join(p.cmdline()):
                procs.append(p)
        except psutil.AccessDenied:
            pass
        except psutil.ZombieProcess:
            pass
    return procs


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    Context manager.
    
    Redirects a file object or file descriptor to a new file descriptor.
    
    Example:
    with open('my-stdout.txt', 'w') as f:
        with stdout_redirected(f):
            print('Writing to my-stdout.txt')
    
    Motivation: In pure-Python, you can redirect all print() statements like this:
        
        sys.stdout = open('myfile.txt')
        
        ...but that doesn't redirect any compiled printf() (or std::cout) output
        from C/C++ extension modules.

    This context manager uses a superior approach, based on low-level Unix file
    descriptors, which redirects both Python AND C/C++ output.
    
    Lifted from the following link (with minor edits):
    https://stackoverflow.com/a/22434262/162094
    (MIT License)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)

    try:
        if fileno(to) == stdout_fd:
            # Nothing to do; early return
            yield stdout
            return
    except ValueError: # filename
        pass

    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        flush(stdout)  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            flush(stdout)
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def flush(stream):
    try:
        libc.fflush(None) # Flush all C stdio buffers
        stream.flush()
    except (AttributeError, ValueError, IOError):
        pass # unsupported


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def tee_streams(output_path, append=False):
    """
    Context manager.
    All stdout and stderr will be tee'd to a a file on disk.
    (in addition to appearing on the original stdout streams).
    
    Note: Stdout and stderr will be merged, in both the tee file and the console.
    """
    if append:
        append = '-a'
    else:
        append = ''
    
    tee = Popen(f'tee {append} {output_path}', shell=True, stdin=PIPE,
                bufsize=1, universal_newlines=True, # line buffering
                preexec_fn=os.setpgrp)  # Spawn the tee process in its own process group,
                                        # so it won't receive SIGINT.
                                        # (Otherwise it might close its input stream too early if the user hits Ctrl+C.)
    try:
        try:
            with stdout_redirected(tee.stdin, stdout=sys.stdout): # pipe stdout to tee
                with stdout_redirected(sys.stdout, stdout=sys.stderr): # merge stderr into stdout
                    yield
        finally:
            tee.stdin.close()
            try:
                tee.wait(1.0)
            except TimeoutExpired:
                pass
    except:
        # If an exception was raised, append the traceback to the file
        with open(output_path, 'a') as f:
            traceback.print_exc(file=f)
        raise


JANELIA_GANGLIA = "cganglia.int.janelia.org/ganglia"
def construct_ganglia_link(hosts, from_timestamp, to_timestamp=None, ganglia_server=JANELIA_GANGLIA):
    if isinstance(hosts, str):
        hosts = [hosts]
    
    if isinstance(from_timestamp, str):
        from_timestamp = parse_timestamp(from_timestamp)

    if isinstance(to_timestamp, str):
        to_timestamp = parse_timestamp(to_timestamp)

    def format_ts(ts):
        if ts is None:
            return ''
        assert isinstance(ts, datetime.datetime)
        year, month, day, hour, minute, *_ = ts.timetuple()
        return f'{month:02}/{day:02}/{year}+{hour:02}:{minute:02}'

    cs = format_ts(from_timestamp)
    ce = format_ts(to_timestamp)
    host_str = '|'.join(hosts)

    url = f'http://{ganglia_server}/?r=custom&cs={cs}&ce={ce}&m=load_one&tab=ch&vn=&hide-hf=false&hreg[]={host_str}'
    return url


@contextmanager
def email_on_exit(email_config, workflow_name, execution_dir, logpath):
    """
    Context manager.
    
    Sends an email when the context exits with success/fail status in the subject line.
    Doesn't work unless sendmail() works without a password
    (i.e. probably won't work on your laptop, will work on a Janelia cluster node).
    
    Args:
        email_config:
            See flyemflows.workflow.base.base_schemas.ExitEmailSchema
        
        workflow_name:
            Name of the workflow class to be reported in the email.
        
        execution_dir:
            Location of the workflow config/data files to be reported in the email.
        
        logpath:
            Location of the logfile whose contents will be included in
            the email if email_config["include-log"] is True.
        
    """
    if not email_config["send"]:
        yield
        return
    
    if not email_config["addresses"]:
        logger.warning("Your config enabled the exit-email feature, but "
                       "no email addresses were listed. Nothing will be sent.")
        yield
        return
    
    user = getpass.getuser()
    host = socket.gethostname()
    jobname = os.environ.get("LSB_JOBNAME", None)

    addresses = []
    for address in email_config["addresses"]:
        if address == "JANELIA_USER":
            address = f'{user}@janelia.hhmi.org'
        addresses.append(address)

    with Timer() as timer:
        def send_email(headline, result, error_str=None):
            body = (headline +
                    f"Duration: {timer.timedelta}\n"
                    f"Execution directory: {execution_dir}\n")

            if jobname:
                body += f"Job name: {jobname}\n"

            if error_str:
                body += f"Error: {error_str}\n"

            if email_config["include-log"]:
                # Sync first, in the hope that the log will flush to disk before we read it.
                # Note:
                #    Currently raised exceptions haven't been printed yet,
                #    so they aren't yet in the log file in your email.
                #    They'll only be present in the on-disk logfile.
                try:
                    # This can hang, apparently.
                    # Hangs like this might be fairly damaging, unfortunately.
                    # According to Ken:
                    #   >If sync is trying to write a file down to disk that was deleted,
                    #   >it can hang like that. Unfortunately, the node will have to be
                    #   >power cycled to deal with this situation.
                    #
                    # Let's hope that's not common.
                    # We'll just timeout the ordinary way and hope for the best.
                    subprocess_run("sync", timeout=10.0)
                    time.sleep(2.0)
                except TimeoutExpired:
                    logger.warning("Timed out while waiting for filesystem sync")

                body += "\nLOG (possibly truncated):\n\n"
                with open(f'{logpath}', 'r') as log:
                    body += log.read()

            msg = MIMEText(body)
            msg['Subject'] = f'Workflow exited: {result}'
            msg['From'] = f'flyemflows <{user}@{host}>'
            msg['To'] = ','.join(addresses)

            try:
                s = smtplib.SMTP('mail.hhmi.org')
                s.sendmail(msg['From'], addresses, msg.as_string())
                s.quit()
            except:
                msg = ("Failed to send completion email.  Perhaps your machine "
                "is not configured to send login-less email, which is required for this feature.")
                logger.error(msg)

        try:
            yield
        except BaseException as ex:
            send_email(f"Workflow {workflow_name} failed: {type(ex)}\n", 'FAILED', str(ex))
            raise
        else:
            send_email(f"Workflow {workflow_name} exited successfully.\n", 'success')


