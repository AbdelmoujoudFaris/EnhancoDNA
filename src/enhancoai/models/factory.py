"""Model construction from configuration (section 41-42: training GUI/CLI model selection)."""

from __future__ import annotations

import torch.nn as nn

from enhancoai.utils.config import ModelConfig
from enhancoai.models.cnn3d import CNN3DModel
from enhancoai.models.gnn import GNNModel
from enhancoai.models.temporal import TemporalModel
from enhancoai.models.hybrid import HybridCooperativityModel, HybridModelConfig

ARCHITECTURES = {"cnn3d", "gnn", "temporal", "hybrid"}


def build_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "cnn3d":
        return CNN3DModel(embedding_dim=config.embedding_dim)
    if config.architecture == "gnn":
        return GNNModel(embedding_dim=config.embedding_dim)
    if config.architecture == "temporal":
        return TemporalModel(embedding_dim=config.embedding_dim)
    if config.architecture == "hybrid":
        hybrid_config = HybridModelConfig(embedding_dim=config.embedding_dim, tasks=config.tasks)
        return HybridCooperativityModel(hybrid_config)
    raise ValueError(f"Unknown model architecture '{config.architecture}'. Choose from {sorted(ARCHITECTURES)}.")
