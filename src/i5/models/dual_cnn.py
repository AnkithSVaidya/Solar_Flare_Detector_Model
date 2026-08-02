import torch
import torch.nn as nn

class DualCNNFlareClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # ===== CONTINUUM BRANCH =====
        self.cont_conv1 = nn.Conv2d(1, 8, 3, padding=1, padding_mode="reflect")
        self.cont_bn1 = nn.BatchNorm2d(8)
        self.cont_conv2 = nn.Conv2d(8, 16, 3, padding=1, padding_mode="reflect")
        self.cont_bn2 = nn.BatchNorm2d(16)

        # ===== MAGNETOGRAM BRANCH =====
        self.mag_conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.mag_bn1 = nn.BatchNorm2d(8)
        self.mag_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.mag_bn2 = nn.BatchNorm2d(16)
        self.mag_conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.mag_bn3 = nn.BatchNorm2d(32)
        self.mag_conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.mag_bn4 = nn.BatchNorm2d(64)

        # Shared components
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool2d(2)

        # ===== FUSION LAYER =====
        # Continuum: 16*32*32 = 16384
        # Magnetogram: 64*8*8 = 4096 (after 4 pools)
        self.fc_cont = nn.Linear(16*32*32, 32)
        self.fc_mag = nn.Linear(64*8*8, 32)
        
        # Fuse both branches
        self.fc_fusion1 = nn.Linear(64, 32)  # 32 + 32 = 64
        self.fc_fusion2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, continuum, magnetogram):
        # ===== CONTINUUM BRANCH =====
        x_cont = self.pool(self.relu(self.cont_bn1(self.cont_conv1(continuum))))
        x_cont = self.pool(self.relu(self.cont_bn2(self.cont_conv2(x_cont))))
        x_cont = x_cont.view(x_cont.size(0), -1)
        x_cont = self.relu(self.fc_cont(x_cont))  # [batch, 32]

        # ===== MAGNETOGRAM BRANCH =====
        x_mag = self.pool(self.leaky_relu(self.mag_bn1(self.mag_conv1(magnetogram))))
        x_mag = self.pool(self.leaky_relu(self.mag_bn2(self.mag_conv2(x_mag))))
        x_mag = self.pool(self.leaky_relu(self.mag_bn3(self.mag_conv3(x_mag))))
        x_mag = self.pool(self.leaky_relu(self.mag_bn4(self.mag_conv4(x_mag))))
        x_mag = x_mag.view(x_mag.size(0), -1)
        x_mag = self.leaky_relu(self.fc_mag(x_mag))  # [batch, 32]

        # ===== FUSION =====
        combined = torch.cat([x_cont, x_mag], dim=1)  # [batch, 64]
        x = self.dropout(combined)
        x = self.relu(self.fc_fusion1(x))
        x = self.fc_fusion2(x)
        
        return x

# Usage
model = DualCNNFlareClassifier()

continuum = torch.randn(32, 1, 128, 128)
magnetogram = torch.randn(32, 1, 128, 128)

output = model(continuum, magnetogram)
print(output.shape)  # [32, 1]