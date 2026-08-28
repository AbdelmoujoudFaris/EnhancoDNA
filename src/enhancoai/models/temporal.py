"""Model 3: temporal MD network -- a Transformer encoder over per-frame features (section 26)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from enhancoai.md.trajectory_features import FEATURE_COLUMNS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalEncoder(nn.Module):
    """(B, T, F) per-frame feature sequence -> embedding vector."""

    def __init__(
        self,
        n_features: int = len(FEATURE_COLUMNS),
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        embedding_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embedding = nn.Linear(d_model, embedding_dim)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(frame_features)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.embedding(pooled)


class TemporalModel(nn.Module):
    """Standalone temporal network with a cooperativity classification head."""

    def __init__(self, n_features: int = len(FEATURE_COLUMNS), embedding_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.encoder = TemporalEncoder(n_features=n_features, embedding_dim=embedding_dim, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, 1))

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(frame_features)
        return self.head(embedding).squeeze(-1)
