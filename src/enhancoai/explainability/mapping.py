"""Map AI attribution scores onto structure entities for 3D visualisation (section 30)."""

from __future__ import annotations

import pandas as pd
import torch


def node_scores_to_frame(node_ids: list[str], scores: torch.Tensor) -> pd.DataFrame:
    """(chain, res_seq, importance) table from graph node ids + a per-node score tensor."""
    scores_list = scores.flatten().detach().cpu().tolist()
    rows = []
    for node_id, score in zip(node_ids, scores_list):
        chain_id, res_seq = node_id.split(":")
        rows.append({"chain_id": chain_id, "res_seq": int(res_seq), "importance": float(score)})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        span = frame["importance"].max() - frame["importance"].min()
        frame["importance_normalised"] = (
            (frame["importance"] - frame["importance"].min()) / span if span > 1e-12 else 0.0
        )
    return frame


def ranked_entities(frame: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Rank | Entity | Position | Importance table (section 29)."""
    ranked = frame.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
