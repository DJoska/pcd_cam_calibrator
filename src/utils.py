import re
import pickle

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

def load_pickle(pickle_file):
    """
    Loads a dictionary from a saved skeleton .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    return(data)

def save_pickle(object, pickle_file):
    """
    Saves a given object to a pickle file at the given filepath
    """
    with open(pickle_file, 'wb') as f:
        pickle.dump(object, f)
    print("Saved ", pickle_file)
    
