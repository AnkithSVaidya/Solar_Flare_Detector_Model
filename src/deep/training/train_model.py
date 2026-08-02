import torch
import torch.nn as nn
from preprocessing.flare_dataset import FlareDataset

from models.magnetogram_cnn import MagnetogramCNN
from models.continuum_cnn import ContinuumCNN
from models.uv_cnn import UVCNN

from constants import *

import matplotlib.pyplot as plt
def train_model(relevant_feature = "magnetogram"):

    # Load the model
    if relevant_feature == "magnetogram":
        model = MagnetogramCNN()
    elif relevant_feature == "continuum":
        model = ContinuumCNN()
    else:
        model = UVCNN()

    # Tell the model to train on the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the optimizer as Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Define the train, val, and test datasets
    train_dataset = FlareDataset(features=[relevant_feature], type='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4,       # tune based on your CPU core count
        pin_memory=True,     # speeds up CPU->GPU transfer
        persistent_workers=True,  # avoids worker restart overhead each epoch)
    )
    val_dataset = FlareDataset(features=[relevant_feature], type='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    train_ids = set(train_dataset.metadata["id"])
    val_ids = set(val_dataset.metadata["id"])
    overlap = train_ids & val_ids
    print(f"Overlapping samples between train/val: {len(overlap)}")

    # Determine what percentage of samples are flares
    n_pos = train_dataset.metadata["is_flare"].sum()
    n_neg = len(train_dataset.metadata) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)

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
            # X = X.unsqueeze(1)          # (B, H, W) -> (B, 1, H, W): add channel dim for Conv2d
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
            torch.save(model.state_dict(), f"src/deep/trained_models/{relevant_feature}_cnn.pt")


    plt.plot(range(NUM_EPOCHS), train_losses, label="Training Loss")
    plt.plot(range(NUM_EPOCHS), val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
