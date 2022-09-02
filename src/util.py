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
    with open(fn, "rb") as f:
        npzfile = np.load(f, allow_pickle=True)  # dict-like, but only temporary
        data = {"X": npzfile["X"], "y": npzfile["y"]}
    return data


def save_category_data(fn: str, data: dict[str, np.ndarray]) -> None:
    """Save examples and labels to .npz file.

    Args:
        fn: the filepath of the category data

        data: a dict of the form {'X': np.ndarray of shape `(num_examples, num_features)`, 'y': np.ndarray of shape `(num_examples)`}
    """
    with open(fn, "wb") as f:
        np.savez(f, X=data["X"], y=data["y"])
    print(f"Saved category data arrays to {fn}.")

def save_learning_results(fn: str, data: np.ndarray) -> None:
    """Save learning results for each of the NN avg loss on each the categories to a .npy file.

    Args:
        fn: the filepath to save the learning results to

        data: a 2D numpy array of shape `(num_categories, num_learners)`.
    """
    kwargs = {str(i+1): data[i] for i in range(len(data))}

    with open(fn, "wb") as f:
        np.savez(f, **kwargs)
    print(f"Saved learning results arrays to {fn}.")

def load_learning_results(fn: str) -> dict[str, np.ndarray]:
    """Load learning results for each of the NN avg loss on each the categories from a .npy file.

    Args:
        fn: the filepath to load the results from

        data: a dict with category names as keys and a numpy array as values corresponding to the avg losses of NN learners on the category.
    """
    with open(fn, "rb") as f:
        npzfile = np.load(f, allow_pickle=True)
        data = {file: npzfile[file] for file in npzfile.files}
    return data