"""Model 2: protein-DNA graph neural network (section 25).

Uses PyTorch Geometric's message-passing layers when available (preferred,
more efficient); otherwise falls back to a pure-PyTorch scatter-add graph
convolution so the model runs with only ``torch`` installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from enhancoai.graph.features import NODE_FEATURE_DIM

try:
    from torch_geometric.nn import GraphConv as _TGGraphConv

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


class SimpleGraphConv(nn.Module):
    """Minimal message-passing layer: mean-aggregate neighbour features, then a linear transform.

    Used only when PyTorch Geometric is not installed.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_transform = nn.Linear(in_dim, out_dim)
        self.neighbor_transform = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)
        if edge_index.numel() == 0:
            return self.self_transform(x)

        src, dst = edge_index[0], edge_index[1]
        messages = self.neighbor_transform(x[src])
        aggregated = torch.zeros(n_nodes, messages.size(-1), device=x.device, dtype=messages.dtype)
        aggregated.index_add_(0, dst, messages)
        degree = torch.zeros(n_nodes, device=x.device, dtype=messages.dtype)
        degree.index_add_(0, dst, torch.ones(dst.size(0), device=x.device, dtype=messages.dtype))
        degree = degree.clamp(min=1).unsqueeze(-1)
        return self.self_transform(x) + aggregated / degree


class GNNEncoder(nn.Module):
    """Graph -> embedding vector (mean-pooled node embeddings)."""

    def __init__(self, in_dim: int = NODE_FEATURE_DIM, hidden_dim: int = 128, embedding_dim: int = 512, n_layers: int = 3):
        super().__init__()
        layer_cls = _TGGraphConv if _HAS_PYG else SimpleGraphConv
        dims = [in_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([layer_cls(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.embedding = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        if batch is None:
            pooled = x.mean(dim=0, keepdim=True)
        else:
            n_graphs = int(batch.max().item()) + 1
            pooled = torch.zeros(n_graphs, x.size(-1), device=x.device, dtype=x.dtype)
            pooled.index_add_(0, batch, x)
            counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
            pooled = pooled / counts
        return self.embedding(pooled)

    def node_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return per-node hidden representations (used for residue/DNA importance heads)."""
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        return x


class GNNModel(nn.Module):
    """Standalone GNN with a cooperativity classification head."""

    def __init__(self, in_dim: int = NODE_FEATURE_DIM, embedding_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.encoder = GNNEncoder(in_dim=in_dim, embedding_dim=embedding_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        embedding = self.encoder(x, edge_index, batch)
        return self.head(embedding).squeeze(-1)
