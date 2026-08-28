"""Vanilla gradient saliency maps (no external dependency)."""

from __future__ import annotations

import torch


def saliency_map(model: torch.nn.Module, inputs: dict[str, torch.Tensor], output_key: str = "cooperativity_logit") -> torch.Tensor:
    """|d output / d input| for the first tensor-valued input requiring grad.

    Returns a tensor the same shape as that input.
    """
    model.eval()
    target_key = None
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            target_key = key
            break
    if target_key is None:
        raise ValueError("No differentiable (floating-point) tensor input found for saliency computation.")

    tensor = inputs[target_key].clone().detach().requires_grad_(True)
    call_inputs = dict(inputs)
    call_inputs[target_key] = tensor

    outputs = model(**call_inputs)
    output = outputs[output_key] if isinstance(outputs, dict) else outputs
    scalar = output.sum()
    scalar.backward()
    return tensor.grad.detach().abs()
