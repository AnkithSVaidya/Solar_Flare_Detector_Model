"""
Finds and displays misclassified samples (false positives and false negatives)
one at a time, with the raw input, the Grad-CAM heatmap, and key metadata,
so you can inspect *why* the model might be getting them wrong.

Run in your project directory alongside magnetogram_cnn.py, magnetogram_dataset.py,
grad_cam.py, and your saved model weights.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from magnetogram_cnn import MagnetogramCNN
from magnetogram_dataset import MagnetogramDataset
from grad_cam import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MagnetogramCNN().to(device)
model.load_state_dict(torch.load("magnetogram_cnn.pt", map_location=device))
model.eval()

val_dataset = MagnetogramDataset(training=False)


@torch.no_grad()
def get_all_predictions(model, dataset, device, threshold=0.5):
    """Runs the model over the dataset once, returns per-sample prob/pred/label."""
    results = []
    for idx in range(len(dataset)):
        X, y = dataset[idx]
        x_input = X.unsqueeze(0).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(x_input)).item()
        pred = int(prob > threshold)
        label = int(y.item())
        results.append({"idx": idx, "prob": prob, "pred": pred, "label": label})
    return results


def find_misclassified(results):
    """Splits results into false positives and false negatives."""
    fp = [r for r in results if r["pred"] == 1 and r["label"] == 0]
    fn = [r for r in results if r["pred"] == 0 and r["label"] == 1]
    return fp, fn


def show_one_misclassified(model, dataset, sample, device, kind):
    """
    Shows one misclassified sample in detail: raw input, Grad-CAM heatmap,
    and an overlay, with the true label / predicted probability in the title.
    'kind' is just a label for the plot title ("False Positive" / "False Negative").
    """
    idx = sample["idx"]
    X, y = dataset[idx]
    x_input = X.unsqueeze(0).unsqueeze(0).to(device)
    x_input.requires_grad_()

    gradcam = GradCAM(model, model.conv2)
    heatmap = gradcam.generate(x_input)

    original = X.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    
    fig.suptitle(
        f"{kind}  |  idx={idx}  |  true label={sample['label']}  |  "
        f"predicted prob={sample['prob']:.3f}",
        fontsize=12,
    )

    axes[0].imshow(original, cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_title("Input (magnetogram)")
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


def show_misclassified_examples(model, dataset, device, n_fp=3, n_fn=3, sort_by_confidence=True):
    """
    Main entry point. Finds misclassified samples and displays them one at a time.

    sort_by_confidence=True shows the model's most CONFIDENTLY wrong examples first
    (e.g. false positives with prob near 1.0, false negatives with prob near 0.0) --
    these are usually the most informative failures to inspect, since they represent
    cases where the model was not just wrong, but wrong with conviction.
    """
    print("Running model over full validation set...")
    results = get_all_predictions(model, dataset, device)
    fp, fn = find_misclassified(results)

    print(f"Total false positives: {len(fp)}")
    print(f"Total false negatives: {len(fn)}")

    if sort_by_confidence:
        fp = sorted(fp, key=lambda r: -r["prob"])       # most confidently "flare" first
        fn = sorted(fn, key=lambda r: r["prob"])        # most confidently "no flare" first

    print(f"\nShowing {min(n_fp, len(fp))} false positive(s)...")
    for sample in fp[:n_fp]:
        show_one_misclassified(model, dataset, sample, device, "False Positive")

    print(f"\nShowing {min(n_fn, len(fn))} false negative(s)...")
    for sample in fn[:n_fn]:
        show_one_misclassified(model, dataset, sample, device, "False Negative")


if __name__ == "__main__":
    show_misclassified_examples(model, val_dataset, device, n_fp=3, n_fn=3)