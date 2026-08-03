
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from models.magnetogram_cnn import MagnetogramCNN
from models.continuum_cnn import ContinuumCNN
from models.uv_cnn import UVCNN

from preprocessing.flare_dataset import FlareDataset
from evaluation.grad_cam import GradCAM

def evaluate_model(relevant_feature = "magnetogram"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    # Load the model
    if relevant_feature == "magnetogram":
        model = MagnetogramCNN()
    elif relevant_feature == "continuum":
        model = ContinuumCNN()
    else:
        model = UVCNN()

    model = model.to(device)
    model.load_state_dict(torch.load(f"src/deep/trained_models/{relevant_feature}_cnn.pt", map_location=device))
    model.eval()

    # --- Load the TEST dataset (never touched during training/val) ---
    test_dataset = FlareDataset(features=[relevant_feature], type="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


    @torch.no_grad()
    def evaluate(model, loader, device, threshold=0.5):
        all_preds = []
        all_labels = []
        all_probs = []

        for X, y in loader:
            X, y = X.to(device), y.to(device)

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
    print(f"{relevant_feature.capitalize()} Confusion matrix:")
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
        f"{relevant_feature.capitalize()} Confusion Matrix -- Test Set\n"
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
        dataset_id = dataset.metadata.iloc[idx]["id"]

        x_input = X.unsqueeze(0).unsqueeze(0).to(device)
        x_input.requires_grad_()

        gradcam = GradCAM(model, model.conv2)
        heatmap = gradcam.generate(x_input)

        original = X.squeeze(0).cpu().numpy()

        # Check if image has negative values
        has_negative = (original < 0).any()
        input_cmap = 'seismic' if has_negative else 'gray'
        vmin = -1 if has_negative else 0
        vmax = 1

        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        fig.suptitle(
            f"{tag}  |  idx={dataset_id}  |  true label={true_label}  |  "
            f"predicted label={pred_label}  |  predicted prob={prob:.3f}",
            fontsize=11,
        )
        
        axes[0].imshow(original, cmap=input_cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Input {relevant_feature}")
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Grad-CAM heatmap")
        axes[1].axis('off')

        axes[2].imshow(original, cmap='gray')
        axes[2].imshow(heatmap, cmap='jet', alpha=0.25)
        axes[2].set_title("Overlay")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


    # Find all correct and incorrect predictions
    correct_indices = [i for i in range(n) if all_preds[i] == all_labels[i]]
    incorrect_indices = [i for i in range(n) if all_preds[i] != all_labels[i]]

    # Pick random samples
    correct_idx = random.choice(correct_indices) if correct_indices else None
    incorrect_idx = random.choice(incorrect_indices) if incorrect_indices else Non

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