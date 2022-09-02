import sys
import util
import numpy as np
import pandas as pd
import plotnine as pn
from data.categories import categories

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/train.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    # Load configs
    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)
    learning_results_fn = configs["filepaths"]["learning_results"]

    # load Fass and Feldman data
    mdl_complexities = np.array([categories[category]["codelength"] for category in categories])

    # load NN learning data
    learning_data = util.load_learning_results(learning_results_fn)
    # print("LEARNING DATA: ", learning_data)

    category_losses = np.array(list(learning_data.values())) # shape `(num_categories, num_learners)`
    mean_category_losses = np.mean(category_losses)

    df = pd.DataFrame(
        {
            'mdl': mdl_complexities,
            'effort': mean_category_losses,
        }
    )
    print(df)



if __name__ == "__main__":
    main()