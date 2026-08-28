"""Model inference (section 61): never report a bare score, always the full interpretation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from enhancoai.models.factory import build_model
from enhancoai.models.hybrid import DEFAULT_MECHANISM_CLASSES
from enhancoai.training.checkpoints import load_checkpoint
from enhancoai.utils.config import ModelConfig


@dataclass
class Prediction:
    is_cooperative: bool
    probability: float
    cooperativity_strength: float | None
    mechanism: str | None
    mechanism_probabilities: dict[str, float] | None
    confidence: float
    evidence: list[str] = field(default_factory=list)
    important_protein_a_residues: list[str] = field(default_factory=list)
    important_dna_bases: list[str] = field(default_factory=list)
    weights_available: bool = True

    def to_dict(self) -> dict:
        return self.__dict__


class Predictor:
    def __init__(self, checkpoint_path: str | Path, model_config: ModelConfig, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.model = build_model(model_config).to(self.device)
        self.model.eval()
        self.checkpoint_path = Path(checkpoint_path)
        self.weights_available = self.checkpoint_path.exists()
        if self.weights_available:
            load_checkpoint(self.checkpoint_path, self.model, map_location=self.device)

    @torch.no_grad()
    def predict(
        self,
        voxel_grid: torch.Tensor | None = None,
        graph_x: torch.Tensor | None = None,
        graph_edge_index: torch.Tensor | None = None,
        graph_node_ids: list[str] | None = None,
        frame_features: torch.Tensor | None = None,
        top_k_evidence: int = 5,
    ) -> Prediction:
        if not self.weights_available:
            return Prediction(
                is_cooperative=False,
                probability=float("nan"),
                cooperativity_strength=None,
                mechanism=None,
                mechanism_probabilities=None,
                confidence=0.0,
                evidence=["Model weights unavailable. Train on an appropriate dataset (see `enhancoai train`)."],
                weights_available=False,
            )

        kwargs = {}
        if voxel_grid is not None:
            kwargs["voxel_grid"] = voxel_grid.to(self.device)
        if graph_x is not None and graph_edge_index is not None:
            kwargs["graph_x"] = graph_x.to(self.device)
            kwargs["graph_edge_index"] = graph_edge_index.to(self.device)
        if frame_features is not None:
            kwargs["frame_features"] = frame_features.to(self.device)

        outputs = self.model(**kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = {"cooperativity_logit": outputs}

        probability = float("nan")
        is_cooperative = False
        if "cooperativity_logit" in outputs:
            probability = float(torch.sigmoid(outputs["cooperativity_logit"]).flatten()[0].item())
            is_cooperative = probability >= 0.5

        strength = None
        if "cooperativity_strength" in outputs:
            strength = float(outputs["cooperativity_strength"].flatten()[0].item())

        mechanism, mechanism_probs = None, None
        if "mechanism_logits" in outputs:
            probs = torch.softmax(outputs["mechanism_logits"], dim=-1).flatten().cpu().tolist()
            mechanism_probs = dict(zip(DEFAULT_MECHANISM_CLASSES, probs))
            mechanism = max(mechanism_probs, key=mechanism_probs.get)

        confidence = abs(probability - 0.5) * 2 if probability == probability else 0.0  # NaN-safe

        important_residues, important_dna = [], []
        if graph_node_ids is not None:
            if "residue_importance" in outputs:
                important_residues = _top_k_nodes(graph_node_ids, outputs["residue_importance"], top_k_evidence, protein_only=True)
            if "dna_importance" in outputs:
                important_dna = _top_k_nodes(graph_node_ids, outputs["dna_importance"], top_k_evidence, protein_only=False)

        evidence = []
        if mechanism is not None:
            evidence.append(f"Predicted mechanism: {mechanism.replace('_', ' ')}")
        if important_residues:
            evidence.append(f"Top contributing protein residues: {', '.join(important_residues)}")
        if important_dna:
            evidence.append(f"Top contributing DNA positions: {', '.join(important_dna)}")

        return Prediction(
            is_cooperative=is_cooperative,
            probability=probability,
            cooperativity_strength=strength,
            mechanism=mechanism,
            mechanism_probabilities=mechanism_probs,
            confidence=confidence,
            evidence=evidence,
            important_protein_a_residues=important_residues,
            important_dna_bases=important_dna,
            weights_available=True,
        )


def _top_k_nodes(node_ids: list[str], scores: torch.Tensor, k: int, protein_only: bool) -> list[str]:
    from enhancoai.graph.features import DNA_BASES

    scores = scores.flatten().cpu().tolist()
    ranked = sorted(zip(node_ids, scores), key=lambda x: x[1], reverse=True)
    return [node_id for node_id, _ in ranked[:k]]
