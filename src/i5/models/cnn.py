import torch
import torch.nn as nn


class FlareCNN(nn.Module):

    def __init__(self, output_features=256):
        super().__init__()

        self.features = nn.Sequential(

            # [20,128,128]
            nn.Conv2d(
                in_channels=20,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [32,64,64]


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [64,32,32]


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [128,16,16]


            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )


        self.pool = nn.AdaptiveAvgPool2d((1,1))


        self.fc = nn.Linear(
            256,
            output_features
        )


    def forward(self, x):

        x = self.features(x)

        # [B,256,16,16]
        x = self.pool(x)

        # [B,256,1,1]
        x = torch.flatten(
            x,
            start_dim=1
        )

        # [B,256]
        x = self.fc(x)

        return x