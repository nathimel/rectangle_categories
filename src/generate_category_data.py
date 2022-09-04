import copy 
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

    def get_one_hot(indices) -> np.ndarray:
        """Utility function to create a single one-hot (2D) ndarray representing a stimulus (in)consistent with a category."""
        one_hot = np.zeros_like(concept)
        one_hot[tuple(indices)] = 1
        return one_hot

    def concept_to_examples(concept: np.ndarray) -> list:
        """Utility function to create a list of (binary 2D array) examples from a concept (binary 2D array)."""
        return [get_one_hot(indices) for indices in np.argwhere(concept)]

    # generate positive examples
    positive_examples = concept_to_examples(concept)
    positive_labels = [1.0] * len(positive_examples)

    # generate negative examples
    negative_examples = concept_to_examples(1 - concept)
    negative_labels = [0.0] * len(negative_examples)

    data = {
        "X": np.array(positive_examples + negative_examples),  # (16, 4, 4)
        "y": np.array(positive_labels + negative_labels),  # (16,)
    }

    return data


def expand_concept(concept: np.ndarray) -> np.ndarray:
  """Helper function to expand (square stretch) a category of size 4x4 to 28x28."""
  concept_large = np.zeros((28, 28))  
  for i in range(len(concept_large)):
    for j in range(len(concept_large[i])):
      i_ = int(np.floor(i / 7))
      j_ = int(np.floor(j / 7))
      if concept[i_, j_]:
        concept_large[i,j] = 1.0
  return concept_large


def expand_categories(categories: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert list of category concepts from size 4x4 to 28x28.

    Args:
        categories: original dict of categories including concepts and hypotheses of size 4x4.

    Returns:
        categories_large: dict of categories including concepts of size 28x28 (and no other data).
    """
    categories_large = {}
    for category in categories:
        categories_large[category] = copy.deepcopy(categories[category])

        concept = categories[category]['concept']
        concept_large = expand_concept(concept)

        categories_large[category]['concept'] = concept_large

    return categories_large

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/generate_categories.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)
    data_size = configs["data_size"]
    dataset_folder = configs["filepaths"]["datasets"]

    to_fn = lambda name: f"{dataset_folder}{name}.npz" # '/' already included

    if data_size == "large":
        experiment_categories = expand_categories(categories)
    elif data_size == "small":
        experiment_categories = categories
    else:
        raise ValueError(f"The config parameter 'data_size' must be 'large' or 'small'. Received {data_size}.")

    for name, category_data in experiment_categories.items():
        data = concept_to_data(category_data["concept"])
        util.save_category_data(fn=to_fn(name), data=data)


if __name__ == "__main__":
    main()
