"""YAML-backed configuration loading with dataclass validation.

All scientific parameters in EnhancoAI must flow through this module rather
than being hard-coded in analysis functions (see project requirement: no
hidden parameters).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import copy

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class StructureConfig:
    protein_chains: list[str] = field(default_factory=list)
    dna_chains: list[str] = field(default_factory=list)


@dataclass
class ContactsConfig:
    protein_dna_cutoff: float = 5.0
    protein_protein_cutoff: float = 5.0
    hydrogen_bond_distance: float = 3.5
    salt_bridge_distance: float = 4.0


@dataclass
class MDConfig:
    trajectory_stride: int = 1
    block_size: int = 100


@dataclass
class FreeEnergyConfig:
    temperature: float = 300.0
    n_bins: int = 50


@dataclass
class ModelConfig:
    architecture: str = "hybrid"
    embedding_dim: int = 512
    tasks: dict[str, bool] = field(
        default_factory=lambda: {
            "cooperativity": True,
            "mechanism": True,
            "residue_importance": True,
            "dna_importance": True,
        }
    )
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "classification": 1.0,
            "regression": 1.0,
            "mechanism": 1.0,
            "residue": 0.5,
            "dna": 0.5,
        }
    )


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    dropout: float = 0.1
    seed: int = 42
    sequence_identity_cutoff: float = 0.3


@dataclass
class ProjectConfig:
    name: str = "EnhancoAI"
    structure: StructureConfig = field(default_factory=StructureConfig)
    contacts: ContactsConfig = field(default_factory=ContactsConfig)
    md: MDConfig = field(default_factory=MDConfig)
    free_energy: FreeEnergyConfig = field(default_factory=FreeEnergyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path, override_path: str | Path | None = None) -> "ProjectConfig":
        data = load_yaml(path)
        if override_path is not None:
            data = _deep_merge(data, load_yaml(override_path))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        cfg = cls()
        if "project" in data and isinstance(data["project"], dict):
            cfg.name = data["project"].get("name", cfg.name)
        if "structure" in data:
            cfg.structure = StructureConfig(**data["structure"])
        if "contacts" in data:
            cfg.contacts = ContactsConfig(**data["contacts"])
        if "md" in data:
            cfg.md = MDConfig(**data["md"])
        if "free_energy" in data:
            cfg.free_energy = FreeEnergyConfig(**data["free_energy"])
        if "model" in data:
            model_data = dict(data["model"])
            cfg.model = ModelConfig(
                architecture=model_data.get("architecture", cfg.model.architecture),
                embedding_dim=model_data.get("embedding_dim", cfg.model.embedding_dim),
                tasks={**cfg.model.tasks, **model_data.get("tasks", {})},
                loss_weights={**cfg.model.loss_weights, **model_data.get("loss_weights", {})},
            )
        if "training" in data:
            cfg.training = TrainingConfig(**data["training"])
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
