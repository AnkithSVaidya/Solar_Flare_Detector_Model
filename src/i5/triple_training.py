import torch
import torch.nn as nn
from preprocessing.flare_dataset import FlareDataset
from models.triple_cnn import TripleCNNFlareClassifier
from constants import *

import matplotlib.pyplot as plt

def main():

    # Load the model
    model = TripleCNNFlareClassifier()

    # Tell the model to train on the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the optimizer as Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=MAGNETOGRAM_LEARNING_RATE)

    # Define the train, val, and test datasets
    train_dataset = FlareDataset(features=['magnetogram', 'continuum', '304'], type='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataset = FlareDataset(features=['magnetogram', 'continuum', '304'], type='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    
    train_ids = set(train_dataset.metadata["id"])
    val_ids = set(val_dataset.metadata["id"])
    overlap = train_ids & val_ids
    print(f"Overlapping samples between train/val: {len(overlap)}")

    # Determine what percentage of samples are flares
    n_pos = train_dataset.metadata["is_flare"].sum()
    n_neg = len(train_dataset.metadata) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    print("Accuracy if predicted 'all false':", n_neg/n_pos)

    # Define the loss function
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_loss = 10000

    # Store the losses
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):

        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch_data, y in train_loader:
            # batch_data is [batch_size, 3, 128, 128]
            # Split into magnetogram, continuum193, and continuum304
            magnetogram = batch_data[:, 0:1, :, :].to(device)     # [batch_size, 1, 128, 128]
            continuum193 = batch_data[:, 1:2, :, :].to(device)    # [batch_size, 1, 128, 128]
            continuum304 = batch_data[:, 2:3, :, :].to(device)    # [batch_size, 1, 128, 128]
            y = y.to(device).unsqueeze(1)  # (B,) -> (B, 1)

            # Zero the optimizer
            optimizer.zero_grad()
            logits = model(magnetogram, continuum193, continuum304)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * magnetogram.size(0)

        # Training loss
        train_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_data, y in val_loader:
                magnetogram = batch_data[:, 0:1, :, :].to(device)
                continuum193 = batch_data[:, 1:2, :, :].to(device)
                continuum304 = batch_data[:, 2:3, :, :].to(device)
                y = y.to(device).unsqueeze(1)

                logits = model(magnetogram, continuum193, continuum304)
                loss = loss_fn(logits, y)
                val_loss += loss.item() * magnetogram.size(0)

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
            torch.save(model.state_dict(), "triple_cnn.pt")


    plt.plot(range(NUM_EPOCHS), train_losses, label="Training Loss")
    plt.plot(range(NUM_EPOCHS), val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()