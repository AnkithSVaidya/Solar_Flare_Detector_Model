"""
Loads the saved best dual model checkpoint and evaluates it on the held-out
TEST dataset, reporting accuracy, confusion matrix, and precision/recall/F1.
"""
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from models.dual_cnn import DualCNNFlareClassifier
from preprocessing.flare_dataset import FlareDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load the saved best model ---
model = DualCNNFlareClassifier().to(device)
model.load_state_dict(torch.load("dual_cnn.pt", map_location=device))
model.eval()

# --- Load the TEST dataset ---
test_dataset = FlareDataset(features=['magnetogram', 'continuum'], type='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.3):
    all_preds = []
    all_labels = []
    all_probs = []

    for batch_data, y in loader:
        magnetogram = batch_data[:, 0:1, :, :].to(device)
        continuum = batch_data[:, 1:2, :, :].to(device)
        y = y.to(device)

        logits = model(continuum, magnetogram)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > threshold).long()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.long().cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_labels, all_preds, all_probs


all_labels, all_preds, all_probs = evaluate(model, test_loader, device)

n = len(all_labels)
n_pos = sum(all_labels)
n_neg = n - n_pos
majority_baseline_acc = max(n_pos, n_neg) / n
accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n

cm = confusion_matrix(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)

print(f"Test set size: {n}  (positive/flare: {n_pos}, negative/no-flare: {n_neg})")
print(f"Majority-class baseline accuracy: {majority_baseline_acc:.4f}")
print(f"Model accuracy:                   {accuracy:.4f}")
print()
print("Confusion matrix:")
print(f"                  Predicted No-Flare   Predicted Flare")
print(f"Actual No-Flare   {cm[0][0]:<21}{cm[0][1]}")
print(f"Actual Flare      {cm[1][0]:<21}{cm[1][1]}")
print()
print(f"Precision (flare class): {precision:.4f}   -- of predicted flares, fraction that were real")
print(f"Recall    (flare class): {recall:.4f}   -- of real flares, fraction the model caught")
print(f"F1        (flare class): {f1:.4f}")


# --- Visualize the confusion matrix ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Flare", "Flare"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title(
    f"Confusion Matrix -- Test Set\n"
    f"Accuracy: {accuracy:.3f}  |  Precision: {precision:.3f}  |  "
    f"Recall: {recall:.3f}  |  F1: {f1:.3f}"
)
plt.tight_layout()
plt.show()


# --- Show sample predictions (without Grad-CAM, just input visualization) ---
def show_sample(dataset, idx, true_label, pred_label, prob, tag):
    batch_data, y = dataset[idx]
    
    original_mag = batch_data[0, :, :].cpu().numpy()
    original_cont = batch_data[1, :, :].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"{tag}  |  idx={idx}  |  true label={true_label}  |  "
        f"predicted label={pred_label}  |  predicted prob={prob:.3f}",
        fontsize=11,
    )
    
    # Magnetogram
    axes[0].imshow(original_mag, cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_title("Magnetogram")
    axes[0].axis('off')

    # Continuum
    axes[1].imshow(original_cont, cmap='gray')
    axes[1].set_title("Continuum")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


correct_idx = next(
    i for i in range(n) if all_preds[i] == all_labels[i]
)
incorrect_idx = next(
    i for i in range(n) if all_preds[i] != all_labels[i]
)

print(f"\nShowing samples for one correct prediction (idx={correct_idx}) "
      f"and one incorrect prediction (idx={incorrect_idx})...")

show_sample(
    test_dataset, correct_idx,
    true_label=all_labels[correct_idx], pred_label=all_preds[correct_idx],
    prob=all_probs[correct_idx], tag="Correct prediction",
)

show_sample(
    test_dataset, incorrect_idx,
    true_label=all_labels[incorrect_idx], pred_label=all_preds[incorrect_idx],
    prob=all_probs[incorrect_idx], tag="Incorrect prediction",
)