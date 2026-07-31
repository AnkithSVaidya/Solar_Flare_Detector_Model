import torch
import torch.nn as nn
from magnetogram_dataset import MagnetogramDataset
from grad_cam import *
from magnetogram_cnn import MagnetogramCNN
model = MagnetogramCNN()


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_dataset = MagnetogramDataset(training=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# Compute this from your actual training set
n_pos = train_dataset.metadata["is_flare"].sum()
n_neg = len(train_dataset.metadata) - n_pos
pos_weight = torch.tensor([n_neg / n_pos]).to(device)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

val_dataset = MagnetogramDataset(training=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

n_epochs = 20

for epoch in range(n_epochs):
    # --- Training ---
    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        X = X.unsqueeze(1)          # (B, H, W) -> (B, 1, H, W): add channel dim for Conv2d
        y = y.unsqueeze(1)          # (B,) -> (B, 1): match logits shape from fc2

        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)

    train_loss /= len(train_dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            X = X.unsqueeze(1)
            y = y.unsqueeze(1)

            logits = model(X)
            loss = loss_fn(logits, y)
            val_loss += loss.item() * X.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()

    val_loss /= len(val_dataset)
    val_acc = correct / len(val_dataset)

    print(f"Epoch {epoch+1}/{n_epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")
    model.eval()
    visualize_gradcam(model, val_dataset, idx=5, device=device)   # pick any validation sample index
    torch.save(model.state_dict(), "magnetogram_cnn.pt")