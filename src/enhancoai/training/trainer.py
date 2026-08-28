"""Generic multi-task training loop (section 41).

Model-agnostic: works with any of the four EnhancoAI architectures as long
as the DataLoader yields batches whose keys match the model's forward()
signature, plus label keys consumed by :func:`multi_task_loss`.
Supports pause/resume/stop via :class:`TrainingControl`, which the GUI's
training page (or a CLI Ctrl-C handler) can flip; the same object also
drives an experiment directory populated per section 43.
"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from enhancoai.training.checkpoints import save_checkpoint
from enhancoai.training.losses import LossWeights, multi_task_loss
from enhancoai.training.metrics import classification_metrics, regression_metrics
from enhancoai.utils.logging import get_logger
from enhancoai.utils.reproducibility import ReproducibilityRecord, set_global_seed

logger = get_logger(__name__)


@dataclass
class TrainingControl:
    """Thread-safe pause/resume/stop flags for the GUI training page."""

    _pause: threading.Event = field(default_factory=threading.Event)
    _stop: threading.Event = field(default_factory=threading.Event)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()

    def is_paused(self) -> bool:
        return self._pause.is_set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def wait_if_paused(self, poll_seconds: float = 0.2) -> None:
        while self.is_paused() and not self.should_stop():
            self._pause.wait(poll_seconds)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        loss_weights: LossWeights | None = None,
        experiment_dir: str | Path | None = None,
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_weights = loss_weights or LossWeights()
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        self.control = TrainingControl()
        self.history: list[dict] = []
        set_global_seed(seed)
        self.seed = seed

        if self.experiment_dir is not None:
            (self.experiment_dir / "figures").mkdir(parents=True, exist_ok=True)

    def _forward(self, batch: dict) -> dict[str, torch.Tensor]:
        model_kwargs = {
            k: v
            for k, v in batch.items()
            if k in ("voxel_grid", "graph_x", "graph_edge_index", "graph_batch", "frame_features")
        }
        output = self.model(**model_kwargs) if model_kwargs else self.model(batch["input"])
        if isinstance(output, torch.Tensor):
            return {"cooperativity_logit": output}
        return output

    def train_one_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in loader:
            self.control.wait_if_paused()
            if self.control.should_stop():
                break
            batch = _move_batch(batch, self.device)
            self.optimizer.zero_grad()
            outputs = self._forward(batch)
            loss, components = multi_task_loss(outputs, batch, self.loss_weights)
            loss.backward()
            self.optimizer.step()
            for k, v in components.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_probs, all_labels = [], []
        all_reg_pred, all_reg_true = [], []
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in loader:
            batch = _move_batch(batch, self.device)
            outputs = self._forward(batch)
            loss, components = multi_task_loss(outputs, batch, self.loss_weights)
            for k, v in components.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

            if "cooperativity_logit" in outputs and "cooperative_label" in batch:
                all_probs.append(torch.sigmoid(outputs["cooperativity_logit"]).cpu().numpy())
                all_labels.append(batch["cooperative_label"].cpu().numpy())
            if "cooperativity_strength" in outputs and "cooperativity_value" in batch:
                all_reg_pred.append(outputs["cooperativity_strength"].cpu().numpy())
                all_reg_true.append(batch["cooperativity_value"].cpu().numpy())

        metrics = {f"val_{k}": v / max(n_batches, 1) for k, v in totals.items()}
        if all_probs:
            import numpy as np

            metrics.update(
                {f"val_{k}": v for k, v in classification_metrics(np.concatenate(all_labels), np.concatenate(all_probs)).items()}
            )
        if all_reg_pred:
            import numpy as np

            metrics.update(
                {f"val_{k}": v for k, v in regression_metrics(np.concatenate(all_reg_true), np.concatenate(all_reg_pred)).items()}
            )
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None, epochs: int, config_dict: dict | None = None) -> list[dict]:
        if self.experiment_dir is not None and config_dict is not None:
            with open(self.experiment_dir / "config.yaml", "w", encoding="utf-8") as fh:
                import yaml

                yaml.safe_dump(config_dict, fh, sort_keys=False)

        for epoch in range(epochs):
            if self.control.should_stop():
                logger.info("Training stopped at epoch %d by user request.", epoch)
                break
            train_metrics = self.train_one_epoch(train_loader)
            record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
            if val_loader is not None:
                record.update(self.evaluate(val_loader))
            self.history.append(record)
            logger.info("Epoch %d/%d: %s", epoch + 1, epochs, record)

            if self.experiment_dir is not None:
                self._checkpoint(epoch)
                self._write_history()

        return self.history

    def _checkpoint(self, epoch: int) -> None:
        save_checkpoint(self.experiment_dir / "checkpoint.pt", self.model, self.optimizer, epoch)

    def _write_history(self) -> None:
        if not self.history:
            return
        with open(self.experiment_dir / "history.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for row in self.history for k in row}))
            writer.writeheader()
            writer.writerows(self.history)
        with open(self.experiment_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(self.history[-1], fh, indent=2)
