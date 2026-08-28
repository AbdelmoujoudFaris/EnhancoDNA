"""Model 1: 3D CNN over the voxelised protein-DNA interface (section 24)."""

from __future__ import annotations

import torch
import torch.nn as nn

from enhancoai.voxel.representation import N_CHANNELS


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CNN3DEncoder(nn.Module):
    """Voxel grid -> embedding vector."""

    def __init__(self, in_channels: int = N_CHANNELS, base_channels: int = 32, embedding_dim: int = 512, n_residual_blocks: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock3D(base_channels * 2) for _ in range(n_residual_blocks)]
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.embedding = nn.Linear(base_channels * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.conv2(x)
        x = self.residual_blocks(x)
        x = self.global_pool(x).flatten(1)
        return self.embedding(x)


class CNN3DModel(nn.Module):
    """Standalone 3D CNN with a cooperativity classification head."""

    def __init__(self, in_channels: int = N_CHANNELS, embedding_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.encoder = CNN3DEncoder(in_channels=in_channels, embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, voxel_grid: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(voxel_grid)
        return self.head(embedding).squeeze(-1)
