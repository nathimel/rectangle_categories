import sys
import util
from typing import Any

import numpy as np
from data.categories import categories


def generate_data(categories: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Given a sample of (12) categories, generate positive and negative example data for each category, and combine into one dataset.

    Args:
        categories: a dict with category names as keys and numpy arrays representing Minimum Description Length (MDL) Concepts as values.

    Returns:
        data: a dict of the form {'X': np.ndarray of shape `(num_examples, num_features)`, 'y': np.ndarray of shape `(num_examples)`}
    """
    X = []
    y = []
    for _, category_data in categories.items():
        concept = category_data["concept"]
        concept_data = concept_to_data(concept)
        X.append(concept_data["X"])
        y.append(concept_data["y"])

    data = {"X": X, "y": y}
    # print(f"SHAPE OF X in DATA: {np.shape(data['X'])}")
    # print(f"SHAPE OF y in DATA: {np.shape(data['y'])}")
    return data


def concept_to_data(concept: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Given a concept (category) as a 2D bit array, generate a list of arrays representing positive and negative examples (vectors).

    Args:
        concept: a binary 2D numpy array with 1s specifying the extension of the category (concept), and 0s elsewhere.
    Returns:
        data: a dict of the form {'X': np.ndarray of shape `(num_examples, num_features)`, 'y': np.ndarray of shape `(num_examples)`}
    """

    def get_one_hot(index) -> np.ndarray:
        """Utility function to create a single one-hot vector representing a stimulus (in)consistent with a category."""
        one_hot = np.zeros_like(concept.flatten())
        one_hot[index] = 1
        return one_hot

    def concept_to_examples(concept: np.ndarray) -> list:
        """Utility function to create a list of examples from a concept (binary 2D array)."""
        return [get_one_hot(idx) for idx in np.argwhere(concept.flatten())]

    # generate positive examples
    positive_examples = concept_to_examples(concept)
    positive_labels = [1.0] * len(positive_examples)

    # generate negative examples
    negative_examples = concept_to_examples(1 - concept)
    negative_labels = [0.0] * len(negative_examples)

    data = {
        "X": np.array(positive_examples + negative_examples),  # shape (16,16)
        "y": np.array(positive_labels + negative_labels),  # shape (16,)
    }
    return data


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/generate_categories.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)
    dataset_folder = configs["filepaths"]["datasets"]

    to_fn = lambda name: f"{dataset_folder}/{name}.npz"

    for name, category_data in categories.items():
        data = concept_to_data(category_data["concept"])
        util.save_category_data(fn=to_fn(name), data=data)


if __name__ == "__main__":
    main()
