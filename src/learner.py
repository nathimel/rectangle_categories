import torch
from torch import nn

# Tutorial net
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# rectangle experiment net
class Net0(nn.Module):
    def __init__(self) -> None:
        super(Net0, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4 * 4, 16),  # input dim
            nn.ReLU(),
            nn.Linear(16, 16), # hidden layer 1
            nn.ReLU(),
            nn.Linear(16, 16), # hidden layer 2
            nn.ReLU(),            
            nn.Linear(16, 1),  # binary classify into 0,1
        )

    def forward(self, x):
        x = self.flatten(x)
        logit = self.linear_relu_stack(x)
        output = torch.sigmoid(logit)
        return output

class Net1(nn.Module):
    def __init__(self) -> None:
        super(Net1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4 * 4, 64),  # input dim
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),                                    
            nn.Linear(64, 1),  # binary classify into 0,1
        )

    def forward(self, x):
        x = self.flatten(x)
        logit = self.linear_relu_stack(x)
        output = torch.sigmoid(logit)
        return output
