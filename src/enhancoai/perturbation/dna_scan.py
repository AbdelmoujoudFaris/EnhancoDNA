"""DNA base substitution scanning (section 32) via AI perturbation."""

from __future__ import annotations

import pandas as pd
import torch

from enhancoai.perturbation.residue_scan import METHOD_LABEL, _mutate_node
from enhancoai.inference.predictor import Predictor

BASE_SUBSTITUTIONS = {"DA": ["DC", "DG", "DT"], "DC": ["DA", "DG", "DT"], "DG": ["DA", "DC", "DT"], "DT": ["DA", "DC", "DG"]}


def dna_base_scan(
    predictor: Predictor,
    graph_x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    node_ids: list[str],
    chain_id: str,
    original_bases: dict[int, str],
) -> pd.DataFrame:
    """For every DNA position of ``chain_id``, try each biologically valid
    substitution and record the predicted-cooperativity delta.

    ``original_bases`` maps res_seq -> current base name ("DA"/"DC"/"DG"/"DT").
    """
    baseline = predictor.predict(graph_x=graph_x, graph_edge_index=graph_edge_index, graph_node_ids=node_ids)

    rows = []
    for i, node_id in enumerate(node_ids):
        node_chain, res_seq_str = node_id.split(":")
        res_seq = int(res_seq_str)
        if node_chain != chain_id or res_seq not in original_bases:
            continue
        current_base = original_bases[res_seq]
        for substitute in BASE_SUBSTITUTIONS.get(current_base, []):
            mutated_x = _mutate_node(graph_x, i, substitute, is_protein=False)
            mutated_pred = predictor.predict(graph_x=mutated_x, graph_edge_index=graph_edge_index, graph_node_ids=node_ids)
            delta = None
            if baseline.probability == baseline.probability and mutated_pred.probability == mutated_pred.probability:
                delta = mutated_pred.probability - baseline.probability
            rows.append(
                {
                    "chain_id": chain_id,
                    "res_seq": res_seq,
                    "mutation": f"{current_base}{res_seq} -> {substitute}",
                    "baseline_probability": baseline.probability,
                    "mutant_probability": mutated_pred.probability,
                    "delta": delta,
                    "confidence": mutated_pred.confidence,
                    "method": METHOD_LABEL,
                }
            )
    return pd.DataFrame(rows).sort_values("delta") if rows else pd.DataFrame(rows)
