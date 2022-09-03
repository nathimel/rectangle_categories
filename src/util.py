import os
import yaml
import numpy as np
import pandas as pd
from plotnine import ggplot

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def set_seed(seed: int) -> None:
    """Sets random seeds."""
    np.random.seed(seed)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def make_path(fn: str) -> None:
    """Creates the path recursively if it does not exist."""
    dirname = os.path.dirname(fn)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created folder {dirname}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Configs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def load_configs(fn: str) -> dict:
    """Load the configs .yml file as a dict."""
    with open(fn, "r") as stream:
        configs = yaml.safe_load(stream)
    return configs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
    kwargs = {str(i + 1): data[i] for i in range(len(data))}

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


def save_analysis(fn: str, spearman_result) -> None:
    """Save the statistical analysis data (currently just spearman rho and p-value) to a csv file.
    """
    df = pd.DataFrame(data=[("spearman rho", spearman_result.correlation), ("p-value", spearman_result.pvalue)])
    df.to_csv(fn, index=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_plot(fn: str, plot: ggplot, width=10, height=10, dpi=300) -> None:
    """Save a plot with some default settings."""
    plot.save(fn, width=10, height=10, dpi=300)
