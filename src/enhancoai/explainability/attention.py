"""Attention-weight extraction from the hybrid model's cross-modal attention and
the temporal encoder's Transformer, and GNN edge-importance via gradients.
"""

from __future__ import annotations

import torch

MODALITY_NAMES = ["structural", "graph", "temporal"]


def cross_modal_attention_weights(model, voxel_grid=None, graph_x=None, graph_edge_index=None, frame_features=None) -> torch.Tensor:
    """Extract the (3, 3) modality-to-modality attention matrix from the hybrid model."""
    captured = {}

    def hook(_module, _input, output):
        # nn.MultiheadAttention.forward returns (attn_output, attn_weights) by default.
        captured["weights"] = output[1]

    handle = model.cross_modal_attention.attention.register_forward_hook(hook)
    try:
        model(voxel_grid=voxel_grid, graph_x=graph_x, graph_edge_index=graph_edge_index, frame_features=frame_features)
    finally:
        handle.remove()
    return captured.get("weights")


def gnn_edge_importance(model, x: torch.Tensor, edge_index: torch.Tensor, output_key: str = "cooperativity_logit") -> torch.Tensor:
    """Gradient-based edge importance: |d output / d edge_attr-equivalent| proxy.

    Since the pure-PyTorch GNN aggregates via unweighted mean, edge
    importance is estimated by the gradient sensitivity of the output to
    each edge's endpoint node features, aggregated per edge.
    """
    x = x.clone().detach().requires_grad_(True)
    outputs = model(x=x, edge_index=edge_index) if hasattr(model, "forward") else model(x, edge_index)
    output = outputs[output_key] if isinstance(outputs, dict) else outputs
    output.sum().backward()

    node_importance = x.grad.detach().abs().sum(dim=-1)  # (n_nodes,)
    src, dst = edge_index[0], edge_index[1]
    edge_scores = (node_importance[src] + node_importance[dst]) / 2.0
    return edge_scores
