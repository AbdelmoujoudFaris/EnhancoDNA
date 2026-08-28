"""Communication network construction: protein residues + DNA nucleotides as nodes.

Edges combine (any subset of): physical contact, hydrogen bonds, dynamic
cross-correlation and mutual information, each independently weighted and
documented so results are traceable to a specific evidence type (section 20).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


def _node_id(chain_id: str, res_seq: int) -> str:
    return f"{chain_id}:{res_seq}"


def build_communication_graph(
    contact_map: pd.DataFrame | None = None,
    correlation_matrix: pd.DataFrame | None = None,
    correlation_chain_id: str | None = None,
    mutual_information_matrix: pd.DataFrame | None = None,
    mi_chain_id: str | None = None,
    contact_weight: float = 1.0,
    correlation_weight: float = 1.0,
    mi_weight: float = 1.0,
    correlation_threshold: float = 0.3,
) -> nx.Graph:
    """Build a weighted communication graph from any combination of evidence sources.

    ``contact_map`` is expected in the ``residue_contact_map`` format
    (columns chain_a, res_seq_a, chain_b, res_seq_b, min_distance) produced
    by :mod:`enhancoai.interactions.contact_maps`.

    ``correlation_matrix`` / ``mutual_information_matrix`` are square
    DataFrames indexed and columned by residue id within a single chain
    (as produced by :mod:`enhancoai.md.correlations`); ``correlation_chain_id``
    / ``mi_chain_id`` label those residues in the combined graph.
    """
    graph = nx.Graph()

    if contact_map is not None and not contact_map.empty:
        for row in contact_map.itertuples():
            u = _node_id(row.chain_a, row.res_seq_a)
            v = _node_id(row.chain_b, row.res_seq_b)
            weight = contact_weight / (row.min_distance + 1e-6)
            _add_or_strengthen_edge(graph, u, v, weight, "contact")

    if correlation_matrix is not None and correlation_chain_id is not None:
        ids = correlation_matrix.index.tolist()
        for i, res_i in enumerate(ids):
            for res_j in ids[i + 1 :]:
                corr = correlation_matrix.loc[res_i, res_j]
                if abs(corr) >= correlation_threshold:
                    u = _node_id(correlation_chain_id, res_i)
                    v = _node_id(correlation_chain_id, res_j)
                    _add_or_strengthen_edge(graph, u, v, correlation_weight * abs(corr), "correlation")

    if mutual_information_matrix is not None and mi_chain_id is not None:
        ids = mutual_information_matrix.index.tolist()
        max_mi = float(np.nanmax(mutual_information_matrix.to_numpy())) or 1.0
        for i, res_i in enumerate(ids):
            for res_j in ids[i + 1 :]:
                mi = mutual_information_matrix.loc[res_i, res_j]
                if mi > 0:
                    u = _node_id(mi_chain_id, res_i)
                    v = _node_id(mi_chain_id, res_j)
                    _add_or_strengthen_edge(graph, u, v, mi_weight * (mi / max_mi), "mutual_information")

    return graph


def _add_or_strengthen_edge(graph: nx.Graph, u: str, v: str, weight: float, evidence: str) -> None:
    if graph.has_edge(u, v):
        graph[u][v]["weight"] += weight
        graph[u][v]["evidence"].add(evidence)
    else:
        graph.add_edge(u, v, weight=weight, evidence={evidence})


def graph_summary(graph: nx.Graph) -> dict:
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "is_connected": nx.is_connected(graph) if graph.number_of_nodes() > 0 else False,
        "n_connected_components": nx.number_connected_components(graph),
    }
