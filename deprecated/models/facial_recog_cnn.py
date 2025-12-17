import torch
import torch.nn as nn
import torch.nn.functional as F
from ...models.residual_block import ResidualBlock

class EvenBetterNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional/Batch Normalization layers with residual connections
        self.l1 = ResidualBlock(3, 32)
        self.l2 = ResidualBlock(32, 64, stride=2)
        self.l3 = ResidualBlock(64, 128, stride=2)

        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x