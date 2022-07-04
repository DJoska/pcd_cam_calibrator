import re

def natural_sort(l):
    """
    Natural sort function for filenames
    
    Returns:
    l:  array-like
        The sorted array
    ----
    Parameters
    ----
    l:  array-like
        The array to be sorted

    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)