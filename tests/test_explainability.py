import torch

from enhancoai.models.cnn3d import CNN3DModel
from enhancoai.models.gnn import GNNModel
from enhancoai.explainability.saliency import saliency_map
from enhancoai.explainability.integrated_gradients import integrated_gradients
from enhancoai.explainability.gradcam import grad_cam_3d
from enhancoai.explainability.attention import gnn_edge_importance
from enhancoai.explainability.mapping import node_scores_to_frame, ranked_entities
from enhancoai.voxel.representation import N_CHANNELS
from enhancoai.graph.features import NODE_FEATURE_DIM


def test_saliency_map_matches_input_shape():
    model = CNN3DModel(embedding_dim=16)
    x = torch.rand(1, N_CHANNELS, 8, 8, 8)
    saliency = saliency_map(model, {"voxel_grid": x})
    assert saliency.shape == x.shape
    assert (saliency >= 0).all()


def test_integrated_gradients_matches_input_shape():
    model = CNN3DModel(embedding_dim=16)
    x = torch.rand(1, N_CHANNELS, 8, 8, 8)
    attributions = integrated_gradients(model, {"voxel_grid": x}, target_key="voxel_grid", n_steps=5)
    assert attributions.shape == x.shape


def test_grad_cam_output_shape():
    model = CNN3DModel(embedding_dim=16)
    x = torch.rand(1, N_CHANNELS, 8, 8, 8)
    cam = grad_cam_3d(model, x)
    assert cam.dim() == 4
    assert (cam >= 0).all() and (cam <= 1).all()


def test_gnn_edge_importance_length():
    model = GNNModel(embedding_dim=16)
    x = torch.rand(6, NODE_FEATURE_DIM)
    edge_index = torch.randint(0, 6, (2, 10))
    importance = gnn_edge_importance(model, x, edge_index)
    assert importance.shape[0] == edge_index.shape[1]


def test_node_scores_to_frame_and_ranking():
    node_ids = ["A:1", "A:2", "C:3"]
    scores = torch.tensor([0.1, 0.9, 0.5])
    frame = node_scores_to_frame(node_ids, scores)
    assert len(frame) == 3
    assert "importance_normalised" in frame.columns
    ranked = ranked_entities(frame, top_k=2)
    assert len(ranked) == 2
    assert ranked.iloc[0]["chain_id"] == "A" and ranked.iloc[0]["res_seq"] == 2
