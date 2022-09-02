import sys
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import learner
import util


def train(dataloader, model, loss_fn, optimizer) -> float:
    """Train the NN and return the average loss value over all epochs.
    """
    running_loss = 0

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.int64)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1).float()) # expand y to shape [batch_size, 1]

        # record loss
        running_loss += loss.item() * X.size(0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # calculate avg loss over an epoch
    running_loss /= len(dataloader.sampler)
    return running_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.int64)
            pred = model(X)
            test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def get_dataloader(fn: str, batch_size: int) -> DataLoader:
    """Get the pytorch DataLoader for the data contained at a filepath specifying the data generated for a category.
    """
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
    ) -> np.ndarray:
    """Train one or more NNs and return their avg losses over epochs as a measure of learning effort.

    Args:
        train_dataloader: 

        num_learners: 

        epochs: 

        lr: 

    TODO: parallelize
    """

    # For each learner in sample:
    avg_losses = []
    for learner_num in range(num_learners):

        # Initialize model and parameters
        model = learner.Net1().to(device)
        print(model)
        loss_fn = nn.functional.binary_cross_entropy_with_logits
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # Main training loop
        running_loss = 0
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            running_loss += train(train_dataloader, model, loss_fn, optimizer)
            test(train_dataloader, model, loss_fn)
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
    results_fn = configs["filepaths"]["learning_results"]

    # For each category, construct a dataset (loader), and train a sample of neural learners on it.
    dataloaders = [
        get_dataloader(
            fn=f"{dataset_folder}/{i}.npz", # clean this up
            batch_size=batch_size
            ) for i in range(1, 13)
        ]

    # train each learner and collect their avg losses
    kwargs = {"num_learners": sample_size, "epochs": epochs, "lr": lr}
    category_results = [
        train_learners(loader, **kwargs)
        for loader in dataloaders
    ]

    # print("category results: ", category_results)

    # save NN category learning results
    util.save_learning_results(results_fn, category_results)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    main()
