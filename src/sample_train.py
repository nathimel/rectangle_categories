import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import plotnine as pn
import learner
import util
import main_experiment


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/sample_train.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    # Load configs
    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)
    learner_class = configs["learner"]
    epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]
    lr = float(configs["learning_rate"])
    dataset_folder = configs["filepaths"]["datasets"]
    sample_loss_fn = configs["filepaths"]["sample_loss"]
    sample_acc_fn = configs["filepaths"]["sample_accuracy"]
    verbose = configs["verbose"]

    # For each category, construct a dataset (loader), and train a sample of neural learners on it.
    dataloaders = [
        main_experiment.get_dataloader(
            fn=f"{dataset_folder}/{i}.npz", batch_size=batch_size  # clean this up
        )
        for i in range(1, 13)
    ]

    # Record the loss evolution for one learner on each category for inspection
    print("Training one learner on all categories for sample loss trajectories.")
    model = learner.learners[learner_class]().to(main_experiment.device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    category_losses = []
    category_accuracies = []

    for train_dataloader in dataloaders:
        losses = []
        accuracies = []
        for t in tqdm(range(epochs)):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")
            losses.append(
                main_experiment.train(
                    train_dataloader, model, loss_fn, optimizer, verbose
                )
            )
            accuracies.append(
                main_experiment.test(train_dataloader, model, loss_fn, verbose)
            )
        if verbose:
            print("Done!")
        category_losses.append(losses)
        category_accuracies.append(accuracies)

    # create dataframe and plot to visualize losses
    points = [
        (
            epoch,
            category_losses[category][epoch],
            category_accuracies[category][epoch],
            str(category + 1),
        )
        for category in range(len(dataloaders))
        for epoch in range(len(category_losses[category]))
    ]

    df = pd.DataFrame(
        points,
        columns=[
            "Epoch",
            "Loss",
            "Accuracy",
            "Concept",
        ],
    )
    df_categorical = df.assign(
        Concept=pd.Categorical(
            df["Concept"], categories=[str(i + 1) for i in range(len(category_losses))]
        )
    )  # preserve order in legend

    loss_plot = (
        pn.ggplot(data=df_categorical, mapping=pn.aes(x="Epoch", y="Loss"))
        + pn.geom_line(pn.aes(color="Concept"))
        + pn.xlab("Epoch")
        + pn.ylab("Loss")
    )
    acc_plot = (
        pn.ggplot(data=df_categorical, mapping=pn.aes(x="Epoch", y="Accuracy"))
        + pn.geom_line(pn.aes(color="Concept"))
        + pn.ylim(0.7, 1.0)
        + pn.xlab("Epoch")
        + pn.ylab("Accuracy")
    )
    util.save_plot(sample_loss_fn, loss_plot)
    util.save_plot(sample_acc_fn, acc_plot)
    if verbose:
        print("LOSS DATAFRAME: \n-------------------------------")
    if verbose:
        print(df)


if __name__ == "__main__":
    main()
