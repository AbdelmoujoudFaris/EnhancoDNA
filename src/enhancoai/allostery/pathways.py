"""Shortest-path communication pathway discovery and ranking (section 47)."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass
class CommunicationPathway:
    nodes: list[str]
    strength: float  # normalised path strength in [0, 1]; higher = stronger candidate pathway
    length: int


def _distance_graph(graph: nx.Graph) -> nx.Graph:
    """Convert edge weights (strength, higher = stronger) into distances for shortest-path search."""
    distance_graph = nx.Graph()
    distance_graph.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        weight = max(data.get("weight", 1e-6), 1e-6)
        distance_graph.add_edge(u, v, distance=1.0 / weight, weight=weight)
    return distance_graph


def find_pathway(graph: nx.Graph, source: str, target: str) -> CommunicationPathway | None:
    """Shortest weighted path between two nodes (e.g. a TF-A residue and a TF-B residue)."""
    if source not in graph or target not in graph:
        return None
    distance_graph = _distance_graph(graph)
    try:
        nodes = nx.shortest_path(distance_graph, source, target, weight="distance")
    except nx.NetworkXNoPath:
        return None

    edge_weights = [graph[u][v]["weight"] for u, v in zip(nodes[:-1], nodes[1:])]
    strength = float(min(edge_weights) / (max(edge_weights) + min(edge_weights))) if edge_weights else 0.0
    # bottleneck-style strength: weakest link dominates, normalised to [0, 1]
    strength = float(min(edge_weights)) if edge_weights else 0.0
    max_possible = max((d.get("weight", 0) for _, _, d in graph.edges(data=True)), default=1.0) or 1.0
    strength = min(strength / max_possible, 1.0)

    return CommunicationPathway(nodes=nodes, strength=strength, length=len(nodes) - 1)


def rank_pathways(
    graph: nx.Graph,
    sources: list[str],
    targets: list[str],
    top_k: int = 10,
) -> list[CommunicationPathway]:
    """Find and rank pathways between every source/target pair by strength."""
    pathways = []
    for source in sources:
        for target in targets:
            if source == target:
                continue
            pathway = find_pathway(graph, source, target)
            if pathway is not None:
                pathways.append(pathway)
    pathways.sort(key=lambda p: p.strength, reverse=True)
    return pathways[:top_k]


def betweenness_centrality(graph: nx.Graph) -> dict[str, float]:
    return nx.betweenness_centrality(graph, weight=lambda u, v, d: 1.0 / max(d.get("weight", 1e-6), 1e-6))


def eigenvector_centrality(graph: nx.Graph, max_iter: int = 1000) -> dict[str, float]:
    try:
        return nx.eigenvector_centrality(graph, weight="weight", max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        return {node: float("nan") for node in graph.nodes}
