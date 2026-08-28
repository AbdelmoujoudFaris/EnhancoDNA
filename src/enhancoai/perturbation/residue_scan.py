"""Alanine scanning of protein interface residues (section 32) via AI perturbation."""

from __future__ import annotations

import pandas as pd
import torch

from enhancoai.graph.features import node_feature_vector
from enhancoai.inference.predictor import Predictor

METHOD_LABEL = "AI-based in-silico perturbation"


def _mutate_node(graph_x: torch.Tensor, node_index: int, new_res_name: str, is_protein: bool) -> torch.Tensor:
    mutated = graph_x.clone()
    mutated[node_index] = torch.from_numpy(node_feature_vector(new_res_name, is_protein)).to(graph_x.dtype)
    return mutated


def alanine_scan(
    predictor: Predictor,
    graph_x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    node_ids: list[str],
    chain_id: str,
) -> pd.DataFrame:
    """For every interface residue of ``chain_id``, mutate it to ALA and record the
    predicted-cooperativity delta.
    """
    baseline = predictor.predict(graph_x=graph_x, graph_edge_index=graph_edge_index, graph_node_ids=node_ids)

    rows = []
    for i, node_id in enumerate(node_ids):
        node_chain, res_seq = node_id.split(":")
        if node_chain != chain_id:
            continue
        mutated_x = _mutate_node(graph_x, i, "ALA", is_protein=True)
        mutated_pred = predictor.predict(graph_x=mutated_x, graph_edge_index=graph_edge_index, graph_node_ids=node_ids)
        delta = None
        if baseline.probability == baseline.probability and mutated_pred.probability == mutated_pred.probability:
            delta = mutated_pred.probability - baseline.probability
        rows.append(
            {
                "chain_id": chain_id,
                "res_seq": int(res_seq),
                "mutation": f"{node_id} -> ALA",
                "baseline_probability": baseline.probability,
                "mutant_probability": mutated_pred.probability,
                "delta": delta,
                "confidence": mutated_pred.confidence,
                "method": METHOD_LABEL,
            }
        )
    return pd.DataFrame(rows).sort_values("delta") if rows else pd.DataFrame(rows)
