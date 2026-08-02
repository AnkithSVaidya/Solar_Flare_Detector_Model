import torch
import torch.nn as nn
from preprocessing.magnetogram_dataset import MagnetogramDataset

class MagnetogramCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input (32, 1, 128, 128)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        # Second convolution layer
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Third convolution layer
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Fourth convolution layer
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(4*32*32, 32)
        self.fc2 = nn.Linear(32, 1)


    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
