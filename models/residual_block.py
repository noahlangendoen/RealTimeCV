import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inChannels, outChannels, stride=1):
        super().__init__()

        # First convolutional and batch norm layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outChannels),
        )
        
        # Second convolutional and batch normalization layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outChannels)
        )

        # If the input and output channels are not the same, or a stride that is not one is passed, use a 1x1 convolution to match the dimensions
        if inChannels != outChannels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outChannels)
            )
        else:
        # If the input and output channels are the same, or stride is 1, there is no need to change dimensions
            self.skip = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.relu(out)  
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out