"""Community structure detection over the allosteric communication graph."""

from __future__ import annotations

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


def detect_communities(graph: nx.Graph) -> list[set[str]]:
    """Greedy modularity-maximisation community detection (deterministic, no extra dependency)."""
    if graph.number_of_edges() == 0:
        return [{node} for node in graph.nodes]
    communities = greedy_modularity_communities(graph, weight="weight")
    return [set(c) for c in communities]


def modularity(graph: nx.Graph, communities: list[set[str]]) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    return nx.algorithms.community.modularity(graph, communities, weight="weight")
