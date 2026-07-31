import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Hook to grab the forward activations of the target layer
        target_layer.register_forward_hook(self._save_activations)
        # Hook to grab the gradients flowing back into it
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x):
        """
        x: single input tensor, shape (1, 1, H, W)
        Returns: heatmap resized to (H, W), values in [0, 1]
        """
        self.model.eval()
        logit = self.model(x)                     # forward pass, triggers forward hook
        self.model.zero_grad()
        logit.backward()                           # backward pass, triggers backward hook

        # Global-average-pool the gradients over spatial dims -> one weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted sum of activation channels
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # only care about features that positively influenced the prediction

        # Resize to input resolution
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0,1] for visualization
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualize_gradcam(model, dataset, idx, device):
    X, y = dataset[idx]
    x_input = X.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    x_input.requires_grad_()

    gradcam = GradCAM(model, model.conv2)   # target the last conv layer
    heatmap = gradcam.generate(x_input)

    original = X.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_title(f"Input (label={y.item():.0f})")
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
    return heatmap