"""Multi-task loss (section 28).

Ltotal = lambda1 * L_classification + lambda2 * L_regression
       + lambda3 * L_mechanism + lambda4 * L_residue + lambda5 * L_dna

Every lambda is configurable via ``ModelConfig.loss_weights``. A task
contributes zero loss (and is skipped) when its label is absent from the
batch, so partially-labelled datasets do not crash training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    classification: float = 1.0
    regression: float = 1.0
    mechanism: float = 1.0
    residue: float = 0.5
    dna: float = 0.5


def multi_task_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or LossWeights()
    total = torch.zeros((), device=next(iter(outputs.values())).device)
    components: dict[str, float] = {}

    if "cooperativity_logit" in outputs and "cooperative_label" in batch:
        loss = F.binary_cross_entropy_with_logits(
            outputs["cooperativity_logit"], batch["cooperative_label"].float()
        )
        total = total + weights.classification * loss
        components["classification"] = float(loss.item())

    if "cooperativity_strength" in outputs and "cooperativity_value" in batch:
        loss = F.mse_loss(outputs["cooperativity_strength"], batch["cooperativity_value"].float())
        total = total + weights.regression * loss
        components["regression"] = float(loss.item())

    if "mechanism_logits" in outputs and "mechanism_label" in batch:
        loss = F.cross_entropy(outputs["mechanism_logits"], batch["mechanism_label"].long())
        total = total + weights.mechanism * loss
        components["mechanism"] = float(loss.item())

    if "residue_importance" in outputs and "residue_importance_label" in batch:
        loss = F.mse_loss(outputs["residue_importance"], batch["residue_importance_label"].float())
        total = total + weights.residue * loss
        components["residue"] = float(loss.item())

    if "dna_importance" in outputs and "dna_importance_label" in batch:
        loss = F.mse_loss(outputs["dna_importance"], batch["dna_importance_label"].float())
        total = total + weights.dna * loss
        components["dna"] = float(loss.item())

    components["total"] = float(total.item())
    return total, components
