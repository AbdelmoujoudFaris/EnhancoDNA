import torch
from torch.utils.data import Dataset, DataLoader

from enhancoai.models.cnn3d import CNN3DModel
from enhancoai.training.trainer import Trainer
from enhancoai.training.losses import LossWeights, multi_task_loss
from enhancoai.training.checkpoints import save_checkpoint, load_checkpoint
from enhancoai.training.metrics import classification_metrics, regression_metrics
from enhancoai.voxel.representation import N_CHANNELS

import numpy as np
import pytest


class ToyVoxelDataset(Dataset):
    def __init__(self, n_samples: int = 8):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = float(idx % 2)
        return {
            "voxel_grid": torch.rand(N_CHANNELS, 8, 8, 8),
            "cooperative_label": torch.tensor(label),
            "cooperativity_value": torch.tensor(label),
        }


def test_multi_task_loss_skips_missing_labels():
    outputs = {"cooperativity_logit": torch.tensor([0.2, -0.1])}
    batch = {"cooperative_label": torch.tensor([1.0, 0.0])}
    loss, components = multi_task_loss(outputs, batch, LossWeights())
    assert "classification" in components
    assert "regression" not in components
    assert loss.requires_grad is False or loss.item() >= 0


def test_trainer_one_epoch(tmp_path):
    model = CNN3DModel(embedding_dim=16)
    loader = DataLoader(ToyVoxelDataset(8), batch_size=4)
    trainer = Trainer(model, device=torch.device("cpu"), experiment_dir=tmp_path)
    history = trainer.fit(loader, loader, epochs=1, config_dict={"note": "test"})
    assert len(history) == 1
    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "metrics.json").exists()


def test_checkpoint_round_trip(tmp_path):
    model = CNN3DModel(embedding_dim=16)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, epoch=3)

    model2 = CNN3DModel(embedding_dim=16)
    payload = load_checkpoint(path, model2)
    assert payload["epoch"] == 3
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_load_checkpoint_missing_raises_clear_error(tmp_path):
    model = CNN3DModel(embedding_dim=16)
    with pytest.raises(FileNotFoundError, match="Model weights unavailable"):
        load_checkpoint(tmp_path / "does_not_exist.pt", model)


def test_classification_metrics_keys():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.4, 0.3])
    metrics = classification_metrics(y_true, y_prob)
    assert {"f1", "auroc", "mcc"} <= set(metrics.keys())


def test_regression_metrics_keys():
    metrics = regression_metrics(np.array([0.1, 0.5]), np.array([0.2, 0.4]))
    assert {"mae", "r2"} <= set(metrics.keys())
