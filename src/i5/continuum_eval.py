"""
Loads the saved best model checkpoint and evaluates it on the held-out
TEST dataset (data/test), reporting accuracy, the majority-class baseline,
a confusion matrix, and precision/recall/F1 for the flare class.

Run this in your project directory, alongside magnetogram_cnn.py,
magnetogram_dataset.py, and your saved model weights.

Note: this is meant to be run sparingly -- the test set should be your
final check, not something you re-run after every training tweak (that
would slowly turn it into a second validation set through repeated peeking).
"""
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from models.continuum_cnn import ContinuumCNN
from preprocessing.continuum_dataset import ContinuumDataset
from grad_cam import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load the saved best model ---
model = ContinuumCNN().to(device)
model.load_state_dict(torch.load("continuum_cnn.pt", map_location=device))
model.eval()

# --- Load the TEST dataset (never touched during training/val) ---
test_dataset = ContinuumDataset(training=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    all_preds = []
    all_labels = []
    all_probs = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X = X.unsqueeze(1)   # (B, H, W) -> (B, 1, H, W)

        logits = model(X)
        probs = torch.sigmoid(logits).squeeze(1)   # (B,)
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


# --- Grad-CAM on one correct and one incorrect prediction ---
# NOTE: this relies on test_loader having shuffle=False, so that the order of
# all_labels/all_preds/all_probs matches test_dataset's own indexing exactly
# (list position i == test_dataset[i]).

def show_gradcam_sample(model, dataset, idx, true_label, pred_label, prob, device, tag):
    X, y = dataset[idx]
    x_input = X.unsqueeze(0).unsqueeze(0).to(device)
    x_input.requires_grad_()

    gradcam = GradCAM(model, model.conv2)
    heatmap = gradcam.generate(x_input)

    original = X.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        f"{tag}  |  idx={idx}  |  true label={true_label}  |  "
        f"predicted label={pred_label}  |  predicted prob={prob:.3f}",
        fontsize=11,
    )

    axes[0].imshow(original, cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_title("Input (continuum)")
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis('off')

    axes[2].imshow(original, cmap='gray')
    axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


correct_idx = next(
    i for i in range(n) if all_preds[i] == all_labels[i]
)
incorrect_idx = next(
    i for i in range(n) if all_preds[i] != all_labels[i] and i > 22
)

print(f"\nShowing Grad-CAM for one correct prediction (idx={correct_idx}) "
      f"and one incorrect prediction (idx={incorrect_idx})...")

show_gradcam_sample(
    model, test_dataset, correct_idx,
    true_label=all_labels[correct_idx], pred_label=all_preds[correct_idx],
    prob=all_probs[correct_idx], device=device, tag="Correct prediction",
)

show_gradcam_sample(
    model, test_dataset, incorrect_idx,
    true_label=all_labels[incorrect_idx], pred_label=all_preds[incorrect_idx],
    prob=all_probs[incorrect_idx], device=device, tag="Incorrect prediction",
)