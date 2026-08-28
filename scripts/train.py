#!/usr/bin/env python
"""Train an EnhancoAI model on a dataset built by build_dataset.py or generate_demo_dataset.py.

Example:
    python scripts/generate_demo_dataset.py
    python scripts/train.py --config configs/training.yaml \\
        --dataset-dir data/processed/demo_training --experiment-name demo_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class VoxelDataset:
    """Loads (voxel_grid, cooperative_label, cooperativity_value) tuples saved as .pt files,
    described by a metadata.json in the same directory (section 39 schema)."""

    def __init__(self, dataset_dir: Path, split: str | None = None):
        import torch  # noqa: F401 (imported for clarity; torch is required by callers)

        self.dataset_dir = Path(dataset_dir)
        metadata_path = self.dataset_dir / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as fh:
            self.metadata = json.load(fh)
        if split is not None:
            self.metadata = [m for m in self.metadata if m.get("split", "train") == split]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        import torch

        entry = self.metadata[idx]
        tensor_path = entry.get("tensor_path") or f"{entry['sample_id']}.pt"
        payload = torch.load(self.dataset_dir / Path(tensor_path).name, weights_only=True)
        item = {"voxel_grid": payload["voxel_grid"]}
        if "cooperative_label" in payload:
            item["cooperative_label"] = torch.tensor(payload["cooperative_label"])
        if "cooperativity_value" in payload:
            item["cooperativity_value"] = torch.tensor(payload["cooperativity_value"])
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--experiment-name", default="run_001")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"])
    args = parser.parse_args()

    from torch.utils.data import DataLoader

    from enhancoai.utils.config import ProjectConfig
    from enhancoai.utils.device import detect_device, describe
    from enhancoai.models.factory import build_model
    from enhancoai.training.trainer import Trainer
    from enhancoai.training.losses import LossWeights

    config = ProjectConfig.from_yaml(args.config)
    device_info = detect_device(args.device)
    print(describe(device_info))

    dataset = VoxelDataset(args.dataset_dir)
    if len(dataset) == 0:
        raise SystemExit(f"No samples found in {args.dataset_dir}. Run build_dataset.py or generate_demo_dataset.py first.")

    n_val = max(1, int(0.2 * len(dataset)))
    train_indices = list(range(len(dataset) - n_val))
    val_indices = list(range(len(dataset) - n_val, len(dataset)))

    from torch.utils.data import Subset

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=config.training.batch_size)

    model = build_model(config.model)
    print(f"Model: {config.model.architecture} ({sum(p.numel() for p in model.parameters()):,} parameters)")

    experiment_dir = Path("experiments") / args.experiment_name
    trainer = Trainer(
        model,
        device=device_info.torch_device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        loss_weights=LossWeights(**config.model.loss_weights),
        experiment_dir=experiment_dir,
        seed=config.training.seed,
    )
    trainer.fit(train_loader, val_loader, epochs=config.training.epochs, config_dict=config.to_dict())
    print(f"\nTraining complete. Experiment saved to {experiment_dir}")


if __name__ == "__main__":
    main()
