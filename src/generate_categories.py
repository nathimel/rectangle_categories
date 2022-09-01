
from typing import Any

import numpy as np

class RectangleCategory:

    def __init__(
        self, 
        length: int, 
        width: int, 
        name: str, 
        mdl_concept: np.ndarray, 
        mdl_hypothesis: np.ndarray, 
        mdl_codelength: float
        ) -> None:
        self.length = length
        self.width = width
        self.shape = (self.length, self.width)

        self.name = name
        self.mdl_concept = mdl_concept
        self.mdl_hypothesis = mdl_hypothesis
        self.mdl_codelength = mdl_codelength

        if (self.shape != self.mdl_concept.shape):
            raise Exception(f"Shape of category is {self.shape} but mdl_hypothesis has shape {self.mdl_hypothesis.shape}")

        if (self.shape != self.mdl_hypothesis.shape):
            raise Exception(f"Shape of category is {self.shape} but mdl_concept has shape {self.mdl_concept.shape}")
        

def generate_category_data(categories: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Given a sample of (12) categories, generate positive and negative example data for each category, and combine into one dataset.

    Args:
        categories: a dict with category names (ints cast to strings) as keys and numpy arrays representing Minimum Description Length (MDL) Concepts as values.

    Returns:
        data: a dict of the form {'X': np.ndarray of shape `(num_examples, num_features)`, 'y': np.ndarray of shape `(num_examples)`}
    """
    pass
