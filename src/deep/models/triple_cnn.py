import torch
import torch.nn as nn

class TripleCNNFlareClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # ===== MAGNETOGRAM BRANCH =====
        self.mag_conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.mag_bn1 = nn.BatchNorm2d(8)
        self.mag_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.mag_bn2 = nn.BatchNorm2d(16)
        self.mag_conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.mag_bn3 = nn.BatchNorm2d(32)
        self.mag_conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.mag_bn4 = nn.BatchNorm2d(64)

        # ===== CONTINUUM 193 BRANCH =====
        self.cont193_conv1 = nn.Conv2d(1, 8, 3, padding=1, padding_mode="reflect")
        self.cont193_bn1 = nn.BatchNorm2d(8)
        self.cont193_conv2 = nn.Conv2d(8, 16, 3, padding=1, padding_mode="reflect")
        self.cont193_bn2 = nn.BatchNorm2d(16)

        # ===== CONTINUUM 304 BRANCH =====
        self.cont304_conv1 = nn.Conv2d(1, 8, 3, padding=1, padding_mode="reflect")
        self.cont304_bn1 = nn.BatchNorm2d(8)
        self.cont304_conv2 = nn.Conv2d(8, 16, 3, padding=1, padding_mode="reflect")
        self.cont304_bn2 = nn.BatchNorm2d(16)

        # Shared components
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool2d(2)

        # ===== FUSION LAYER =====
        # Magnetogram: 64*8*8 = 4096
        # Continuum 193: 16*32*32 = 16384
        # Continuum 304: 16*32*32 = 16384
        self.fc_mag = nn.Linear(64*8*8, 32)
        self.fc_cont193 = nn.Linear(16*32*32, 32)
        self.fc_cont304 = nn.Linear(16*32*32, 32)
        
        # Fuse all three branches
        self.fc_fusion1 = nn.Linear(96, 64)  # 32 + 32 + 32 = 96
        self.fc_fusion2 = nn.Linear(64, 32)
        self.fc_fusion3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, magnetogram, continuum193, continuum304):
        # ===== MAGNETOGRAM BRANCH =====
        x_mag = self.pool(self.leaky_relu(self.mag_bn1(self.mag_conv1(magnetogram))))
        x_mag = self.pool(self.leaky_relu(self.mag_bn2(self.mag_conv2(x_mag))))
        x_mag = self.pool(self.leaky_relu(self.mag_bn3(self.mag_conv3(x_mag))))
        x_mag = self.pool(self.leaky_relu(self.mag_bn4(self.mag_conv4(x_mag))))
        x_mag = x_mag.view(x_mag.size(0), -1)
        x_mag = self.leaky_relu(self.fc_mag(x_mag))  # [batch, 32]

        # ===== CONTINUUM 193 BRANCH =====
        x_cont193 = self.pool(self.relu(self.cont193_bn1(self.cont193_conv1(continuum193))))
        x_cont193 = self.pool(self.relu(self.cont193_bn2(self.cont193_conv2(x_cont193))))
        x_cont193 = x_cont193.view(x_cont193.size(0), -1)
        x_cont193 = self.relu(self.fc_cont193(x_cont193))  # [batch, 32]

        # ===== CONTINUUM 304 BRANCH =====
        x_cont304 = self.pool(self.relu(self.cont304_bn1(self.cont304_conv1(continuum304))))
        x_cont304 = self.pool(self.relu(self.cont304_bn2(self.cont304_conv2(x_cont304))))
        x_cont304 = x_cont304.view(x_cont304.size(0), -1)
        x_cont304 = self.relu(self.fc_cont304(x_cont304))  # [batch, 32]

        # ===== FUSION =====
        combined = torch.cat([x_mag, x_cont193, x_cont304], dim=1)  # [batch, 96]
        x = self.dropout(combined)
        x = self.relu(self.fc_fusion1(x))
        x = self.relu(self.fc_fusion2(x))
        x = self.fc_fusion3(x)
        
        return x