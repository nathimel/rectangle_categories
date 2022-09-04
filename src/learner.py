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

class MLPLarge0(nn.Module):
    def __init__(self) -> None:
        super(MLPLarge0, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 100),  # input dim
            nn.ReLU(),
            nn.Linear(100, 100), # hidden layer 1
            nn.ReLU(),
            nn.Linear(100, 100), # hidden layer 2
            nn.ReLU(),            
            nn.Linear(100, 1),  # binary classify into 0,1
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
        self.flatten = nn.Flatten()

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

        x = self.flatten(x) # flatten all dimensions except batch
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
        self.flatten = nn.Flatten()

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

        x = self.flatten(x) # flatten all dimensions except batch
        # print("FLATTEN OUTPUT SHAPE: ", x.size())
        x = self.linear_relu_stack(x)
        x = torch.sigmoid(x)
        return x

class CNN2(nn.Module):
    """Copied from https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
    """
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )

        # flatten input
        self.flatten = nn.Flatten()

        # fully connected layer, output binary classification
        self.out = nn.Linear(32 * 7 * 7, 1)
    def forward(self, x):
        # print("SHAPE OF NET INPUT: ")
        # print(x.size())
        # reshape 2d input to an input volume for convolution
        n, w, h = x.size()
        x = x.reshape(n, 1, w, h)
        # print("SHAPE AFTER RESHAPE: ")
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.flatten(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x


############################################################################
# End network classes
############################################################################

# Store all learner classes in dict with string keys so we can specify from config file
learners = {
    "MLP0": MLP0,
    "MLP1": MLP1,
    "MLPLarge0": MLPLarge0,
    "CNN0": CNN0,
    "CNN1": CNN1,
    "CNN2": CNN2,
}        