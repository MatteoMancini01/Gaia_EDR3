import pickle

def save_pickle(name_file, object, dir='hmc_samples'):
    """
    Save a Python object as a pickle file in the specified directory.

    Parameters:
    ----------
    name_file : str
        The name of the file (without extension) to save the object to.
    object : any
        The Python object to be pickled and saved.
    dir : str, optional
        Directory where the file will be saved. Default is 'posterior_samples'.

    Returns:
    -------
    None
    """
    with open(f"{dir}/{name_file}.pkl", "wb") as f:
        pickle.dump(object, f)


def load_pickle(name_file, dir='hmc_samples'):
    """
    Load a Python object from a pickle file in the specified directory.

    Parameters:
    ----------
    name_file : str
        The name of the file (without extension) to load the object from.
    dir : str, optional
        Directory where the file is located. Default is 'posterior_samples'.

    Returns:
    -------
    object : any
        The Python object loaded from the pickle file.
    """
    with open(f"{dir}/{name_file}.pkl", "rb") as f:
        object = pickle.load(f)

    return object