import torch
from torch import nn

############################################################################
# Feedforward networks
# 
# Flatten the input stimulus from a 2D array to a 1d array, and pass through
# a multilayer perceptron.
############################################################################

class MLP0(nn.Module):
    def __init__(self) -> None:
        super(MLP0, self).__init__()
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

class MLP1(nn.Module):
    def __init__(self) -> None:
        super(MLP1, self).__init__()
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

############################################################################
# Convolutional networks
# 
# We treat the grid on which categories are placed as an image. 
# Then 'images' are of size 4x4x1 (4 wide, 4 high, 1 'grayscale channel')
############################################################################

class CNN0(nn.Module):
    def __init__(self) -> None:
        super(CNN0, self).__init__()
        # input shape: [batch_size, in_channels, H, W]
        # output shape: [batch_size, out_channels, H_out, W_out]
        
        # H_out = W_out = [floor(H + 2*padding - kernel_size) / stride] + 1.
        # input shape is (batch_size, 1, 4, 4)

        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=2,
            padding=1,
            stride=1,
            )
        # so here, H_out = W_out = [(4 + 2*1 - 2) / 1] + 1 = 5
        # yielding shape (batch_size, 1, 5, 5)

        # Pooling is also a filter, and uses the same equation. we have
        # ((width + 2 * padding - filter_size) / stride + 1
        self.pool = nn.MaxPool2d(
            kernel_size=3, # pool of square window size=2
            padding=1,
            stride=2,
            )
        # here H_out = W_out = [(5 + 2*1 - 3) / 2] + 1 = 3
        # yielding shape (batch_size, 1, 3, 3)

        self.conv2 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=2,
            padding=1, 
            stride=1,
        )
        # and here H_out = W_out = [(3 + 2*1 - 2) / 1] + 1 = 4
        # yielding shape (batch_size, 1, 4, 4)

        # apply pooling again H_out = W_out = floor((4 + 2*1 - 3) / 2) + 1 = 2
        # yielding shape (batch_size, 1, 2, 2)

        # flatten input
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1 * 2 * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, x):
        # x = self.pool(nn.functional.relu(self.conv1(x)))
        # x = self.pool(nn.functional.relu(self.conv2(x)))
        # print("INPUT SHAPE: ", x.size())
        x = self.conv1(x)
        # print("CONV1 OUTPUT SHAPE: ", x.size())
        x = nn.functional.relu(x)
        x = self.pool(x)
        # print("POOL OUTPUT SHAPE: ", x.size())

        x = self.conv2(x)
        # print("CONV2 OUTPUT SHAPE: ", x.size())

        x = nn.functional.relu(x)
        x = self.pool(x)
        # print("POOL OUTPUT SHAPE: ", x.size())

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print("FLATTEN OUTPUT SHAPE: ", x.size())
        x = self.linear_relu_stack(x)
        x = torch.sigmoid(x)
        return x

class CNN1(nn.Module):
    def __init__(self) -> None:
        super(CNN1, self).__init__()
        # input shape: [batch_size, in_channels, H, W]
        # output shape: [batch_size, out_channels, H_out, W_out]
        
        # H_out = W_out = [floor(H + 2*padding - kernel_size) / stride] + 1.
        # input shape is (batch_size, 1, 4, 4)

        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=2,
            padding=1,
            stride=1,
            )
        # so here, H_out = W_out = [(4 + 2*1 - 2) / 1] + 1 = 5
        # yielding shape (batch_size, 1, 5, 5)

        self.conv2 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=2,
            padding=1, 
            stride=1,
        )
        # and here H_out = W_out = [(5 + 2*1 - 2) / 1] + 1 = 6
        # yielding shape (batch_size, 1, 4, 4)

        # flatten input
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1 * 6 * 6, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    
    def forward(self, x):
        # x = self.pool(nn.functional.relu(self.conv1(x)))
        # x = self.pool(nn.functional.relu(self.conv2(x)))
        # print("INPUT SHAPE: ", x.size())
        x = self.conv1(x)
        # print("CONV1 OUTPUT SHAPE: ", x.size())
        x = nn.functional.relu(x)
        # x = self.pool(x)
        # print("POOL OUTPUT SHAPE: ", x.size())

        x = self.conv2(x)
        # print("CONV2 OUTPUT SHAPE: ", x.size())

        x = nn.functional.relu(x)
        # x = self.pool(x)
        # print("POOL OUTPUT SHAPE: ", x.size())

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print("FLATTEN OUTPUT SHAPE: ", x.size())
        x = self.linear_relu_stack(x)
        x = torch.sigmoid(x)
        return x

############################################################################
# End network classes
############################################################################

# Store all learner classes in dict with string keys so we can specify from config file
learners = {
    "MLP0": MLP0,
    "MLP1": MLP1,
    "CNN0": CNN0,
    "CNN1": CNN1,
}        