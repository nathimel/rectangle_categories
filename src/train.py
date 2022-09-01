import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import learner
import util

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

##############################################################################
# Main driver code
##############################################################################

def main():

    # Load configs
    configs = util.load_configs
    epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]
    lr = float(configs["learning_rate"])

    # Load data
    raw_data = util.load_category_data()
    X = raw_data["X"]
    y = raw_data["y"]
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    training_data = TensorDataset(tensor_X, tensor_y)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    # Initialize model and parameters
    model = learner.Net0().to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Main training loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        learner.train(train_dataloader, model, loss_fn, optimizer)
    print("Done!")                

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")    
    main()