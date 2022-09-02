import sys
import util
import numpy as np
import pandas as pd
import plotnine as pn
from scipy import stats
from data.categories import categories


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/train.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    # Load configs
    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)
    learning_results_fn = configs["filepaths"]["learning_results"]
    plot_fn = configs["filepaths"]["plot"]

    # load Fass and Feldman data
    mdl_complexities = np.array(
        [categories[category]["codelength"] for category in categories]
    )

    # load NN learning data
    learning_data = util.load_learning_results(learning_results_fn)
    # print("LEARNING DATA: ", learning_data)

    category_losses = [
        losses for losses in learning_data.values()
    ]  # shape `(num_categories, num_learners)`
    # print("CATEGORY LOSSES: ", category_losses)

    # create plot points of (mdl, loss) with number of repeat (mdl, ) points equal to number of learners
    points = [
        (categories[category_name]["codelength"], loss)
        for category_name in categories
        for loss in learning_data[category_name]
    ]

    mean_category_losses = np.mean(category_losses, axis=1)

    # analyze and visualize data
    df = pd.DataFrame(points, columns=["mdl", "effort"])
    df_mean = pd.DataFrame({"mdl": mdl_complexities, "effort": mean_category_losses})

    # Plot MDL complexity vs learning effort
    plot = (
        pn.ggplot(data=df, mapping=pn.aes(x="mdl", y="effort"))
        + pn.geom_point(
            size=2,
            shape="+",
            data=df,
        )
        + pn.geom_point(size=4, shape="o", fill="white", data=df_mean)
        + pn.geom_smooth(size=1, data=df, alpha=0.2, color="red", se=True)
        # + pn.ylim(0, 1)
        + pn.xlab("MDL complexity")
        + pn.ylab("Effort (avg loss) for learner")
        + pn.scale_color_cmap("cividis")
    )
    util.save_plot(plot_fn, plot)

    # Report spearman rank correlation for MDL complexity vs learning effort
    result = stats.spearmanr(mdl_complexities, mean_category_losses)
    print("SPEARMAN RHO: ", result.correlation)


if __name__ == "__main__":
    main()
