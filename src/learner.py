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

class CNN0(nn.Module):
    """Treat the grid on which categories are placed as an image. Then images are of size 4x4x1 (4 wide, 4 high, 1 'grayscale channel')"""
    def __init__(self) -> None:
        super(CNN0, self).__init__()
        # input shape: [batch_size, in_channels, height, width]
        # output shape: [batch_size, out_channels, H_out, W_out]
        # where H_out = floor(height + 2*padding - kernel_size) / stride + 1.
        # our input is (batch_size, 1, 4, 4)

        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=2,
            padding=2, 
            stride=1,
            )
        # so here, H_out = W_out = (4 + 2*2 - 2) / 2 = 6/3 = 2
        # yielding shape (batch_size, 1, 2, 2)

        # Pooling is also a filter, and uses the same equation. we have
        # ((width + 2 * padding - filter_size) / stride + 1
        self.pool = nn.MaxPool2d(
            kernel_size=2, # pool of square window size=2
            padding=2,
            stride=1,
            )
        # here H_out = W_out = (2 + 2*2 - 2) / 2 = 2
        # yielding shape (batch_size, 1, 2, 2)

        self.conv2 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=2,
            padding=2, 
            stride=1,
        )
        # and here H_out = W_out = (2 + 2*2 - 2) / 2 = 2
        # yielding shape (batch_size, 1, 2, 2)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1 * 2 * 2, 16), # input flattened
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.linear_relu_stack(x)
        x = torch.sigmoid(x)
        return x