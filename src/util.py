import yaml
import numpy as np
from typing import Any

def set_seed(seed: int) -> None:
    """Sets random seeds."""
    np.random.seed(seed)


def load_configs(fn: str) -> dict[str, Any]:
    """Load the configs .yml file as a dict."""
    with open(fn, "r") as stream:
        configs = yaml.safe_load(stream)
    return configs

def load_category_data(fn: str) -> dict:
    """Load arrays (examples or labels) from .npy files.

    Args: 
        fn: the filepath to load the category data
    Returns:
        data: a dict of the form {'X': np.ndarray of shape `(num_examples, num_features)`, 'y': np.ndarray of shape `(num_examples)`}
    """
    with open(fn, 'rb') as f:
        npzfile = np.load(f, allow_pickle=True) # dict-like, but only temporary
        data = {'X': npzfile['X'], 'y': npzfile['y']}
    return data

def save_category_data(fn: str, data: dict[str, np.ndarray]) -> None:
    """Save examples and labels to .npz file.

    Args: 
        fn: the filepath of the category data    

        data: a dict of the form {'X': np.ndarray of shape `(num_examples, num_features)`, 'y': np.ndarray of shape `(num_examples)`}
    """
    with open(fn, 'wb') as f:
        np.savez(f, X=data['X'], y=data['y'])
    print(f"Saved category data arrays to {fn}.")
