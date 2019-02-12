import lz4
import numpy as np
from neuclease.util import box_to_slicing
from neuclease.dvid import encode_nonaligned_labelarray_volume, decode_labelarray_volume

COMPRESSION_METHODS = ['gzip_labelarray',
                       #'lz4_labelarray', # Not supported yet -- need a tiny libdvid change
                       #'labelarray',     # Not supported yet -- need a tiny libdvid change
                       'lz4',
                       'lz4_2x',
                       ]


def compress_volume(method, volume, box_zyx):
    """
    Compress the volume using the specified scheme, assuming it resides in the given box.
    If the compression scheme requires encoding the volume in a larger (aligned) box than the one given,
    the returned box will differ from the input box.
    
    Returns:
        box_zyx, encoded_data
    """
    assert method in COMPRESSION_METHODS
    
    if method == 'gzip_labelarray':
        assert volume.dtype == np.uint64
        encoded_box, encoded_data = encode_nonaligned_labelarray_volume(box_zyx[0], volume)
        
        # Convert from memoryview (not pickleable) to bytes
        encoded_data = bytes(encoded_data)
        return encoded_box, encoded_data

    if method == 'lz4':
        volume = np.asarray(volume, order='C')
        encoded = lz4.compress(volume)
        return box_zyx, encoded

    if method == 'lz4_2x':
        volume = np.asarray(volume, order='C')
        encoded = lz4.compress(volume)
        encoded = lz4.compress(encoded)
        return box_zyx, encoded


def uncompress_volume(method, encoded_data, dtype, encoded_box_zyx, box_zyx=None):
    """
    Uncompress the given encoded data using the specified scheme.
    If the data was encoded into a box that is larger than the box of interest,
    specify a separate box_zyx for the subvolume of interest.
    """
    if method == 'gzip_labelarray':
        volume = decode_labelarray_volume(encoded_box_zyx, encoded_data)
    
    if method == 'lz4':
        shape = encoded_box_zyx[1] - encoded_box_zyx[0]
        buf = lz4.uncompress(encoded_data)
        volume = np.frombuffer(buf, dtype).reshape(shape)
    
    if method == 'lz4_2x':
        shape = encoded_box_zyx[1] - encoded_box_zyx[0]
        buf = lz4.uncompress(encoded_data)
        buf = lz4.uncompress(buf)
        volume = np.frombuffer(buf, dtype).reshape(shape)

    if box_zyx is None or (box_zyx == encoded_box_zyx).all():
        return volume
    else:
        assert (box_zyx[0] >= encoded_box_zyx[0]).all() and (box_zyx[1] <= encoded_box_zyx[1]).all(), \
            f"box_zyx ({box_zyx.tolist()}) must be contained within encoded_box_zyx ({encoded_box_zyx.tolist()})"
        vol_box = box_zyx - encoded_box_zyx[0]
        return volume[box_to_slicing(*vol_box)]

