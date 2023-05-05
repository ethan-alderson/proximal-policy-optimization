import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# tentatively, I just stuck a convolutional neural net in here that I wrote previously'
# Note that these dimensions are incorrect
class ConvActor(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.usedropout = True

        # the four convolutional layers of the network
        self.conv1 = nn.Sequential(
            # 1 in 256 features out
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=8, stride=1, padding=2),
            # rectify linear unit, if output is positive it produces the output, or else it produces 0,
            # This is our linearization function
            nn.ReLU(),
            # batch normalization keeps the mean of the layer's inputs 0 to prevent skew
            # shape of normalization should be the same as the number of output channels
            nn.BatchNorm2d(256),
            # Concentrates the outputs in pairs of two, takes the max of each pair, concentrating and shrinking the data
            nn.MaxPool2d(kernel_size=2)
        )

        # note that the # of output channels of one layer matches the # of input channels of the next 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=8, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)
            )

        # drop out layer to reduce model overfitting
        # ONLY USED IN TRAINING
        if (self.usedropout):
            self.dropout = nn.Dropout(0.25);   
        

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
            )

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)
            )

        # takes any parameter at flattens it to one dimension
        self.flatten=nn.Flatten()

        # 4608 is the size of the flattened data, final output is 10, for digits 0-9
        self.linear1 = nn.Linear(in_features=in_dim, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=32)
        self.linear4 = nn.Linear(in_features=32, out_features=out_dim)
    
        # a probability function that outputs the highest float of the final outputs, i.e. the digit it is guessing
        # we comment this out because our cross entropy loss cost function does this already
        # self.output = nn.Softmax(dim = 1)

    # tells the network to move forward with the data, inherited by nn.module
    def forward(self, obs):

        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        x = self.conv1(obs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        logits = self.linear4(x)

        return logits

    def prepToTest(self):
        self.usedropout = False