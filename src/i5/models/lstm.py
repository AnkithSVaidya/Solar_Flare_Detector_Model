import torch.nn as nn
import torch
from .cnn import FlareCNN

class FlareCNNLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = FlareCNN(256)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.classifier = nn.Linear(
            128,
            1
        )


    def forward(self,x):

        # x:
        # [B,4,20,128,128]

        features = []

        for t in range(4):
            features.append(
                self.cnn(x[:,t])
            )

        # [B,4,256]
        features = torch.stack(
            features,
            dim=1
        )

        out,_ = self.lstm(features)

        # last timestep
        out = out[:,-1]

        return torch.sigmoid(
            self.classifier(out)
        )