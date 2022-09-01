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

def load_category_data(fn: str) -> dict[str, np.ndarray]:
    """
    Args: 
        fn: the filepath to load the category data
    Returns:
        data: a dict of the form {'X': np.ndarray of shape `(num_examples, num_features)`, 'y': np.ndarray of shape `(num_examples)` }
    """
    pass

def save_category_data(fn: str, data: dict[str, np.ndarray]) -> None:
    """
    Args: 
        fn: the filepath of the category data    

        data: a dict of the form {'X': np.ndarray of shape `(num_examples, num_features)`, 'y': np.ndarray of shape `(num_examples)` }
    """
    pass
