import numpy as np

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

