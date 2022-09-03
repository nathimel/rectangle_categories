import sys
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import plotnine as pn
import learner
import util


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    verbose: bool = False,
) -> float:
    """Train the NN on a category and return the average loss value over all epochs.

    Args:
        dataloader: a pytorch DataLoader object containing the dataset for one category.

        model: the neural network learner to train

        loss_fn: the loss function to optimize, e.g. BCELoss

        optimizer: the optimization algorithm to use, e.g. SGD, Adam, etc.

        verbose: a bool representing whether to print intermediate information during training.
    """
    running_loss = 0

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)

        # Compute prediction error
        pred = model(X)

        if verbose:
            print("TARGET: ", y.unsqueeze(1))
        if verbose:
            print("PREDICTION: ", pred)

        loss = loss_fn(pred, y.unsqueeze(1))  # expand y to shape [batch_size, 1]

        # record loss
        running_loss += loss.item() * X.size(0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        if verbose:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # calculate avg loss over an epoch
    running_loss /= len(dataloader.sampler)
    return running_loss


def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    verbose: bool = False,
) -> None:
    """Evaluate the neural network learner on a category.

    Args:
        dataloader: a pytorch DataLoader object containing the dataset for one category.

        model: the neural network learner to test

        loss_fn: the loss function to evaluate on, e.g. BCELoss

        verbose: a bool representing whether to print intermediate information during training.
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            correct += (torch.round(pred) == y).type(torch.float).sum().detach().numpy()
    test_loss /= num_batches
    correct /= size
    if verbose:
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )


def get_dataloader(fn: str, batch_size: int) -> DataLoader:
    """Get the pytorch DataLoader for the data contained at a filepath specifying the data generated for a category."""
    raw_data = util.load_category_data(fn=fn)
    X = raw_data["X"]
    y = raw_data["y"]
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    training_data = TensorDataset(tensor_X, tensor_y)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    return train_dataloader


def train_learners(
    train_dataloader: DataLoader,
    num_learners: int = 1,
    epochs: int = 4,
    lr: float = 1e-3,
    category_name: str = None,
    verbose: bool = False,
) -> np.ndarray:
    """Train one or more NNs and return their avg losses over epochs as a measure of learning effort.

    Args:
        train_dataloader: a DataLoader containing the data generated by one category

        num_learners: the number of NNs to train on the category

        epochs: the number of complete passes through the data to train for

        lr: learning rate

        category_name: the name of the category (e.g. "1")

        verbose: a bool representing whether to print intermediate information during training.

    Returns:
        avg_losses: a numpy array of shape `(num_learners)` representing each learner's loss value averaged over epochs and batches.
    """
    if category_name is not None:
        print(f"Training {num_learners} learners on Category {category_name}.")

    # For each learner in sample:
    avg_losses = []
    for learner_num in tqdm(range(num_learners)):

        # Initialize model and parameters
        model = learner.CNN0().to(device)
        if verbose:
            print(model)
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # Main training loop
        running_loss = 0
        for t in range(epochs):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")
            running_loss += train(
                train_dataloader, model, loss_fn, optimizer, verbose=verbose
            )
            test(train_dataloader, model, loss_fn)
        if verbose:
            print(f"Learner {learner_num} done!")

        running_loss /= epochs
        # record the avg loss on the category for one learner
        avg_losses.append(running_loss)
    return np.array(avg_losses)


##############################################################################
# Main driver code
##############################################################################


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/train.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    # Load configs
    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)
    epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]
    sample_size = configs["sample_size"]
    lr = float(configs["learning_rate"])
    dataset_folder = configs["filepaths"]["datasets"]
    sample_loss_fn = configs["filepaths"]["sample_loss"]
    results_fn = configs["filepaths"]["learning_results"]
    verbose = configs["verbose"]

    # For each category, construct a dataset (loader), and train a sample of neural learners on it.
    dataloaders = [
        get_dataloader(
            fn=f"{dataset_folder}/{i}.npz", batch_size=batch_size  # clean this up
        )
        for i in range(1, 13)
    ]

    # Record the loss evolution for one learner on each category for inspection
    print("Training one learner on all categories for sample loss trajectories.")

    model = learner.CNN0().to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    category_losses = []

    for train_dataloader in dataloaders:
        losses = []
        for t in tqdm(range(epochs)):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")
            losses.append(train(train_dataloader, model, loss_fn, optimizer, verbose))
            test(train_dataloader, model, loss_fn, verbose)
        if verbose:
            print("Done!")
        category_losses.append(losses)

    # create dataframe and plot to visualize losses
    points = [
        (
            epoch,
            loss,
            str(category_name + 1),
        )
        for category_name in range(len(category_losses))
        for epoch, loss in enumerate(category_losses[category_name])
    ]
    df = pd.DataFrame(
        points,
        columns=[
            "Epoch",
            "Loss",
            "Concept",
        ],
    )
    df_categorical = df.assign(Concept=pd.Categorical(df['Concept'], categories=[str(i+1) for i in range(len(category_losses))])) # preserve order in legend
    plot = (
        pn.ggplot(
            data=df_categorical, mapping=pn.aes(x="Epoch", y="Loss")
        )
        + pn.geom_line(pn.aes(color="Concept"))
        + pn.xlab("Epoch")
        + pn.ylab("Loss")
    )
    util.save_plot(sample_loss_fn, plot)
    if verbose:
        print("LOSS DATAFRAME: \n-------------------------------")
    if verbose:
        print(df)

    ############################################################################
    # Main learning experiment
    ############################################################################

    # train each learner and collect their avg losses
    kwargs = {"num_learners": sample_size, "epochs": epochs, "lr": lr, "verbose": verbose}
    category_results = [
        train_learners(
            loader,
            **kwargs,
            category_name=str(i+1),
            )
        for i, loader in enumerate(dataloaders)
    ]

    # save NN category learning results
    util.save_learning_results(results_fn, category_results)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    main()
