"""Grad-CAM for the 3D CNN encoder (section 29).

Hooks the last convolutional feature map of :class:`CNN3DEncoder`,
weights each channel by the average gradient of the target output with
respect to that channel, and produces a (D, H, W) heatmap the same
spatial size as the last conv layer's output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from enhancoai.models.cnn3d import CNN3DModel


def grad_cam_3d(model: CNN3DModel, voxel_grid: torch.Tensor) -> torch.Tensor:
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    target_layer = model.encoder.residual_blocks[-1]

    def forward_hook(_module, _input, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(voxel_grid)
        output.sum().backward()

        weights = gradients["value"].mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        cam = F.relu((weights * activations["value"]).sum(dim=1))  # (B, D, H, W)

        cam_min = cam.amin(dim=(1, 2, 3), keepdim=True)
        cam_max = cam.amax(dim=(1, 2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach()
    finally:
        handle_f.remove()
        handle_b.remove()
