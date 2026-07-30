import torch
from torch.utils.data import DataLoader

from flare_loader import FlareDataLoader
from flare_dataset import FlareDataset
from models.lstm import FlareCNNLSTM
import os

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    loader = FlareDataLoader()
    loader.show_class_distribution()
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
        persistent_workers=True,   # avoids re-spawning workers every epoch
        prefetch_factor=4          # each worker preloads more batches ahead
    )

    model = FlareCNNLSTM().to(device)

    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 20

    best_loss = float("inf")

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)
   
            optimizer.zero_grad()

            output = model(X)

            loss = criterion(
                output.squeeze(1),
                y
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print("pred:", output.squeeze(1)[:5].detach().cpu())
            print("true:", y[:5].cpu())
        # Average loss over all batches
        epoch_loss = running_loss / len(train_loader)


        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {epoch_loss:.6f}"
        )
        print(output[:5])

        # Save best model
        if epoch_loss < best_loss:

            best_loss = epoch_loss

            torch.save(
                model.state_dict(),
                "best_flare_model.pth"
            )

            print("Saved new best model!")


    print("Training complete.")


if __name__ == "__main__":
    #main()
    loader = FlareDataLoader()
    row = loader.training_target.iloc[0]
    print(row["dataset_id"], row["is_flare"], row["peak_flux"], row["datetime"])