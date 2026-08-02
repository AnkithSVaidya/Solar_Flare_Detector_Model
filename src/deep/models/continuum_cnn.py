import torch
import torch.nn as nn

class ContinuumCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input (32, 1, 128, 128)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, padding_mode="reflect")
        self.bn1 = nn.BatchNorm2d(8)

        # Second convolution layer
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, padding_mode="reflect")
        self.bn2 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(16*32*32, 32)
        self.fc2 = nn.Linear(32, 1)


    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))


        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
