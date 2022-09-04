import sys
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import learner
import util

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("mps") # need mac os 12.3
print(f"Using {device} device")


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
        X = X.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1))  # expand y to shape [batch_size, 1]

        # record loss
        running_loss += loss.item() * X.size(0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        current = batch * len(X)
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
) -> float:
    """Evaluate the neural network learner on a category (one epoch).

    Args:
        dataloader: a pytorch DataLoader object containing the dataset for one category.

        model: the neural network learner to test

        loss_fn: the loss function to evaluate on, e.g. BCELoss

        verbose: a bool representing whether to print intermediate information during training.

    Returns:
        accuracy: the fraction of category examples correct during evaluation
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, dtype=torch.float) # [batch_size, width, height]
            y = y.to(device, dtype=torch.float) # [batch_size]

            pred = model(X) # [batch_size, 1] we must unsqueeze label shape

            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            correct_instance = (torch.round(pred) == y.unsqueeze(1)).type(torch.float).sum().item()
            correct += correct_instance

    test_loss /= num_batches
    accuracy = correct / size
    if verbose:
        print(
            f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
    return accuracy


def get_dataloader(fn: str, batch_size: int) -> DataLoader:
    """Get the pytorch DataLoader for the data contained at a filepath specifying the data generated for a category."""
    raw_data = util.load_category_data(fn=fn)
    X = raw_data["X"]
    y = raw_data["y"]
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    training_data = TensorDataset(tensor_X, tensor_y)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def train_learners(
    train_dataloader: DataLoader,
    learner_class: nn.Module,
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
        model = learner.learners[learner_class]().to(device)
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
# Main experiment driver code
##############################################################################


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 src/main_experiment.py path_to_config_file")
        raise TypeError(f"Expected {2} arguments but received {len(sys.argv)}.")

    # Load configs
    config_fn = sys.argv[1]
    configs = util.load_configs(config_fn)
    learner_class = configs["learner"]
    seed = configs["random_seed"]
    epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]
    sample_size = configs["sample_size"]
    lr = float(configs["learning_rate"])
    dataset_folder = configs["filepaths"]["datasets"]
    results_fn = configs["filepaths"]["learning_results"]
    verbose = configs["verbose"]

    util.set_seed(seed)

    # For each category, construct a dataset (loader), and train a sample of neural learners on it.
    dataloaders = [
        get_dataloader(
            fn=f"{dataset_folder}{i}.npz", 
            batch_size=batch_size,  # clean this up
        )
        for i in range(1, 13)
    ]

    # train each learner and collect their avg losses
    kwargs = {
        "num_learners": sample_size,
        "epochs": epochs,
        "lr": lr,
        "verbose": verbose,
    }
    category_results = [
        train_learners(
            loader,
            learner_class,
            **kwargs,
            category_name=str(i + 1),
        )
        for i, loader in enumerate(dataloaders)
    ]

    # save NN category learning results
    util.save_learning_results(results_fn, category_results)


if __name__ == "__main__":
    main()
