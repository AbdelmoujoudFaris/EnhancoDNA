"""Model 4: hybrid multi-modal, multi-task cooperativity/allostery model (sections 27-28).

3D structural encoder + graph neural network + temporal MD encoder are
fused with cross-modal (multi-head) attention into a shared latent
representation, which then feeds five task heads. Any subset of the three
input modalities may be omitted at call time (e.g. no trajectory
available) -- missing modalities are replaced by a learned placeholder
embedding rather than crashing, so the model degrades gracefully to
whatever evidence is actually available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from enhancoai.models.cnn3d import CNN3DEncoder
from enhancoai.models.gnn import GNNEncoder
from enhancoai.models.temporal import TemporalEncoder

DEFAULT_MECHANISM_CLASSES = [
    "direct_protein_protein_cooperativity",
    "dna_mediated_allostery",
    "mixed_mechanism",
    "weak_or_no_cooperativity",
]


@dataclass
class HybridModelConfig:
    embedding_dim: int = 512
    dropout: float = 0.1
    n_attention_heads: int = 8
    mechanism_classes: list[str] = field(default_factory=lambda: list(DEFAULT_MECHANISM_CLASSES))
    tasks: dict[str, bool] = field(
        default_factory=lambda: {
            "cooperativity": True,
            "mechanism": True,
            "residue_importance": True,
            "dna_importance": True,
        }
    )


class CrossModalAttention(nn.Module):
    """Multi-head self-attention over the stacked modality embeddings."""

    def __init__(self, embedding_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, modality_embeddings: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(modality_embeddings, modality_embeddings, modality_embeddings)
        return self.norm(modality_embeddings + attended)


class HybridCooperativityModel(nn.Module):
    def __init__(self, config: HybridModelConfig | None = None):
        super().__init__()
        self.config = config or HybridModelConfig()
        dim = self.config.embedding_dim

        self.structural_encoder = CNN3DEncoder(embedding_dim=dim)
        self.graph_encoder = GNNEncoder(embedding_dim=dim)
        self.temporal_encoder = TemporalEncoder(embedding_dim=dim, dropout=self.config.dropout)

        # learned placeholders used when a modality is unavailable for a given sample
        self.missing_structural = nn.Parameter(torch.zeros(dim))
        self.missing_graph = nn.Parameter(torch.zeros(dim))
        self.missing_temporal = nn.Parameter(torch.zeros(dim))

        self.cross_modal_attention = CrossModalAttention(dim, self.config.n_attention_heads, self.config.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.ReLU(inplace=True), nn.Dropout(self.config.dropout)
        )

        self.cooperativity_head = nn.Linear(dim, 1)  # binary logit
        self.strength_head = nn.Linear(dim, 1)  # regression, cooperativity strength in [0, 1] via sigmoid
        self.mechanism_head = nn.Linear(dim, len(self.config.mechanism_classes))
        self.residue_importance_head = nn.Linear(dim, 1)
        self.dna_importance_head = nn.Linear(dim, 1)

    def encode(
        self,
        voxel_grid: torch.Tensor | None = None,
        graph_x: torch.Tensor | None = None,
        graph_edge_index: torch.Tensor | None = None,
        graph_batch: torch.Tensor | None = None,
        frame_features: torch.Tensor | None = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        if voxel_grid is not None:
            structural = self.structural_encoder(voxel_grid)
        else:
            structural = self.missing_structural.unsqueeze(0).expand(batch_size, -1).to(device)

        if graph_x is not None and graph_edge_index is not None:
            graph = self.graph_encoder(graph_x, graph_edge_index, graph_batch)
        else:
            graph = self.missing_graph.unsqueeze(0).expand(batch_size, -1).to(device)

        if frame_features is not None:
            temporal = self.temporal_encoder(frame_features)
        else:
            temporal = self.missing_temporal.unsqueeze(0).expand(batch_size, -1).to(device)

        stacked = torch.stack([structural, graph, temporal], dim=1)  # (B, 3, dim)
        attended = self.cross_modal_attention(stacked)
        fused = self.fusion(attended.flatten(1))
        return fused

    def forward(
        self,
        voxel_grid: torch.Tensor | None = None,
        graph_x: torch.Tensor | None = None,
        graph_edge_index: torch.Tensor | None = None,
        graph_batch: torch.Tensor | None = None,
        frame_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = 1
        for tensor in (voxel_grid, frame_features):
            if tensor is not None:
                batch_size = tensor.size(0)
                break

        latent = self.encode(voxel_grid, graph_x, graph_edge_index, graph_batch, frame_features, batch_size)

        outputs = {"latent": latent}
        if self.config.tasks.get("cooperativity", True):
            outputs["cooperativity_logit"] = self.cooperativity_head(latent).squeeze(-1)
            outputs["cooperativity_strength"] = torch.sigmoid(self.strength_head(latent)).squeeze(-1)
        if self.config.tasks.get("mechanism", True):
            outputs["mechanism_logits"] = self.mechanism_head(latent)
        if graph_x is not None and graph_edge_index is not None and (
            self.config.tasks.get("residue_importance", True) or self.config.tasks.get("dna_importance", True)
        ):
            # project hidden-dim node representations into the shared embedding_dim space
            # so the importance heads (sized for `dim`) can consume them.
            node_embeddings = self.graph_encoder.embedding(
                self.graph_encoder.node_embeddings(graph_x, graph_edge_index)
            )
            if self.config.tasks.get("residue_importance", True):
                outputs["residue_importance"] = self.residue_importance_head(node_embeddings).squeeze(-1)
            if self.config.tasks.get("dna_importance", True):
                outputs["dna_importance"] = self.dna_importance_head(node_embeddings).squeeze(-1)

        return outputs
