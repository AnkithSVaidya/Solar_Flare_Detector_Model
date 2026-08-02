import torch
import torch.nn as nn
from preprocessing.continuum_dataset import ContinuumDataset
from grad_cam import *
from models.continuum_cnn import ContinuumCNN
from constants import *

import matplotlib.pyplot as plt

# Load the model
model = ContinuumCNN()

# Tell the model to train on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the optimizer as Adam
optimizer = torch.optim.Adam(model.parameters(), lr=MAGNETOGRAM_LEARNING_RATE)

# Define the train, val, and test datasets
train_dataset = ContinuumDataset(training=True, validation=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = ContinuumDataset(training=True, validation=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = ContinuumDataset(training=False)
test_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Determine what percentage of samples are flares
n_pos = train_dataset.metadata["is_flare"].sum()
n_neg = len(train_dataset.metadata) - n_pos
pos_weight = torch.tensor([n_neg / n_pos]).to(device)
print("Accuracy if predicted 'all false':", n_neg/n_pos)

# Define the loss function
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
best_loss = 10000               # Define the best loss as a very high number

# Store the losses (to graph afterwards)
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):

    # --- Training ---
    model.train()
    train_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        X = X.unsqueeze(1)          # (B, H, W) -> (B, 1, H, W): add channel dim for Conv2d
        y = y.unsqueeze(1)          # (B,) -> (B, 1): match logits shape from fc2

        # Zero the optimizer
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)

    # Training loss
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

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")
    if best_loss > val_loss:
        print("New Best Model Found!")
        best_loss = val_loss
        torch.save(model.state_dict(), "continuum_cnn.pt")


plt.plot(range(NUM_EPOCHS), train_losses, label="Training Loss")
plt.plot(range(NUM_EPOCHS), val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
