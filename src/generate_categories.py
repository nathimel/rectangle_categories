import sys
import util
from typing import Any

import numpy as np
from data.categories import categories

def generate_category_data(categories: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Given a sample of (12) categories, generate positive and negative example data for each category, and combine into one dataset.

    Args:
        categories: a dict with category names (ints cast to strings) as keys and numpy arrays representing Minimum Description Length (MDL) Concepts as values.

    Returns:
        data: a dict of the form {'X': np.ndarray of shape `(num_examples, num_features)`, 'y': np.ndarray of shape `(num_examples)`}
    """
    X = []
    y = []
    for _, category_data in categories.items():
        concept = category_data["concept"]
        examples = concept_to_examples(concept)

        for example in examples["positives"]:
            X.append(example)
            y.append(1)

        for example in examples["negatives"]:
            X.append(example)
            y.append(0)

    data = {'X': np.array(X), 'y': np.array(y)}
    # print(f"SHAPE OF X in DATA: {np.shape(data['X'])}")
    # print(f"SHAPE OF y in DATA: {np.shape(data['y'])}")
    return data

def concept_to_examples(concept: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Given a concept (category) as a 2D bit array, generate a list of arrays representing positive and negative examples (vectors).

    Args: 
        concept: a binary 2D numpy array with 1s specifying the extension of the category (concept), and 0s elsewhere. 
    Returns:
        examples: a dict of the form {'positives': ..., 'negatives': ...}
    """

    def get_one_hot(index) -> np.ndarray:
        one_hot = np.zeros_like(concept.flatten())
        one_hot[index] = 1
        return one_hot

    # generate positive examples
    argw_flat = np.argwhere(concept.flatten())
    positives = [get_one_hot(idx) for idx in argw_flat]

    # generate negative examples
    neg_concept = 1 - concept
    argw_flat_neg = np.argwhere(neg_concept)
    negatives = [1 - get_one_hot(idx) for idx in argw_flat_neg]

    examples = {"positives": positives, "negatives": negatives}

    return examples


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/generate_categories.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)
    dataset_fn = configs["filepaths"]["dataset"]

    data = generate_category_data(categories=categories)

    util.save_category_data(fn=dataset_fn, data=data)

if __name__ == "__main__":
    main()