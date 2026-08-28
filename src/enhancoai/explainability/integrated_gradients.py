"""Integrated Gradients (Sundararajan et al. 2017), implemented directly on
top of ``torch.autograd`` so no extra dependency (e.g. Captum) is required.
"""

from __future__ import annotations

import torch


def integrated_gradients(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    target_key: str,
    output_key: str = "cooperativity_logit",
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
) -> torch.Tensor:
    """Attribute the model's output to ``inputs[target_key]`` via a straight-line
    path integral from ``baseline`` (default: all-zeros) to the input.
    """
    model.eval()
    tensor = inputs[target_key]
    baseline = baseline if baseline is not None else torch.zeros_like(tensor)

    total_gradients = torch.zeros_like(tensor)
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interpolated = (baseline + alpha * (tensor - baseline)).clone().detach().requires_grad_(True)
        call_inputs = dict(inputs)
        call_inputs[target_key] = interpolated

        outputs = model(**call_inputs)
        output = outputs[output_key] if isinstance(outputs, dict) else outputs
        output.sum().backward()
        total_gradients = total_gradients + interpolated.grad.detach()

    average_gradients = total_gradients / n_steps
    attributions = (tensor - baseline) * average_gradients
    return attributions
