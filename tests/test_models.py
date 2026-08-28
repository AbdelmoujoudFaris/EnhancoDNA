import torch

from enhancoai.models.cnn3d import CNN3DModel
from enhancoai.models.gnn import GNNModel
from enhancoai.models.temporal import TemporalModel
from enhancoai.models.hybrid import HybridCooperativityModel, HybridModelConfig
from enhancoai.models.factory import build_model
from enhancoai.utils.config import ModelConfig
from enhancoai.voxel.representation import N_CHANNELS
from enhancoai.graph.features import NODE_FEATURE_DIM
from enhancoai.md.trajectory_features import FEATURE_COLUMNS


def test_cnn3d_forward_shape():
    model = CNN3DModel(embedding_dim=32)
    x = torch.rand(2, N_CHANNELS, 16, 16, 16)
    out = model(x)
    assert out.shape == (2,)


def test_gnn_forward_shape():
    model = GNNModel(embedding_dim=32)
    x = torch.rand(10, NODE_FEATURE_DIM)
    edge_index = torch.randint(0, 10, (2, 20))
    out = model(x, edge_index)
    assert out.shape == (1,)


def test_gnn_forward_with_batch():
    model = GNNModel(embedding_dim=32)
    x = torch.rand(10, NODE_FEATURE_DIM)
    edge_index = torch.randint(0, 10, (2, 20))
    batch = torch.tensor([0] * 5 + [1] * 5)
    out = model(x, edge_index, batch)
    assert out.shape == (2,)


def test_temporal_forward_shape():
    model = TemporalModel(embedding_dim=32)
    frames = torch.rand(3, 15, len(FEATURE_COLUMNS))
    out = model(frames)
    assert out.shape == (3,)


def test_hybrid_forward_all_modalities():
    model = HybridCooperativityModel(HybridModelConfig(embedding_dim=32))
    voxel = torch.rand(1, N_CHANNELS, 16, 16, 16)
    x = torch.rand(8, NODE_FEATURE_DIM)
    edge_index = torch.randint(0, 8, (2, 16))
    frames = torch.rand(1, 10, len(FEATURE_COLUMNS))
    outputs = model(voxel_grid=voxel, graph_x=x, graph_edge_index=edge_index, frame_features=frames)
    assert outputs["cooperativity_logit"].shape == (1,)
    assert outputs["mechanism_logits"].shape == (1, 4)
    assert outputs["residue_importance"].shape == (8,)
    assert outputs["dna_importance"].shape == (8,)


def test_hybrid_forward_missing_modalities_degrades_gracefully():
    model = HybridCooperativityModel(HybridModelConfig(embedding_dim=16))
    voxel = torch.rand(1, N_CHANNELS, 16, 16, 16)
    outputs = model(voxel_grid=voxel)
    assert outputs["cooperativity_logit"].shape == (1,)
    assert "residue_importance" not in outputs


def test_build_model_factory():
    for architecture in ("cnn3d", "gnn", "temporal", "hybrid"):
        model = build_model(ModelConfig(architecture=architecture, embedding_dim=16))
        assert sum(p.numel() for p in model.parameters()) > 0
