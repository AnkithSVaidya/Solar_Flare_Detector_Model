"""
Single-timestep CNN-only baseline.

Purpose: isolate whether the LSTM/temporal modeling is the source of the
"confident but uncorrelated with truth" predictions, by dropping the
temporal component entirely and training a plain CNN + linear classifier
on ONE timestep of data.

If this baseline ALSO fails to beat ~0.69 loss (random-guess level) or
shows the same "confident but wrong" pattern, the problem is upstream of
the LSTM -- i.e. in the CNN, the input data/features, or normalization.

If this baseline DOES learn (loss drops meaningfully, predictions
correlate with truth), the problem is specific to the LSTM/temporal
stacking in FlareCNNLSTM.forward.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from flare_loader import FlareDataLoader
from flare_dataset import FlareDataset
from models.cnn import FlareCNN


class FlareCNNBaseline(nn.Module):
    """CNN + linear classifier, no LSTM, no temporal stacking."""

    def __init__(self):
        super().__init__()
        self.cnn = FlareCNN(256)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        # x: [B, 20, 128, 128]  (single timestep only)
        features = self.cnn(x)
        return self.classifier(features)  # raw logits -- use BCEWithLogitsLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loader = FlareDataLoader()

    train_dataset = FlareDataset(
        loader.training_target,
        os.path.join(loader.root_data_dir, "training")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    model = FlareCNNBaseline().to(device)

    # Using logits + BCEWithLogitsLoss (more stable than sigmoid + BCELoss)
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        first_batch = True

        for X, y in train_loader:
            # X: [B, 4, 20, 128, 128] -- take ONLY the first timestep
            X = X[:, 0].to(device)   # -> [B, 20, 128, 128]
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits.squeeze(1), y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if first_batch:
                probs = torch.sigmoid(logits.squeeze(1))[:5].detach().cpu()
                print("pred:", probs)
                print("true:", y[:5].cpu())
                first_batch = False

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_baseline_model.pth")
            print("Saved new best model!")

    print("Baseline training complete.")
    model.eval()
    visualize_gradcam(model, val_dataset, idx=5, device=device)   # pick any validation sample index

if __name__ == "__main__":
    main()
