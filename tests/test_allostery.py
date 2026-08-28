import networkx as nx
import pandas as pd

from enhancoai.allostery.network import build_communication_graph, graph_summary
from enhancoai.allostery.pathways import find_pathway, rank_pathways, betweenness_centrality
from enhancoai.allostery.communities import detect_communities, modularity


def _toy_contact_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"chain_a": "A", "res_seq_a": 1, "chain_b": "C", "res_seq_b": 5, "min_distance": 3.0},
            {"chain_a": "A", "res_seq_a": 2, "chain_b": "C", "res_seq_b": 6, "min_distance": 4.0},
        ]
    )


def test_build_communication_graph_from_contacts():
    graph = build_communication_graph(contact_map=_toy_contact_map())
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 2
    summary = graph_summary(graph)
    assert summary["n_nodes"] == 4


def test_find_pathway_between_connected_nodes():
    graph = build_communication_graph(contact_map=_toy_contact_map())
    pathway = find_pathway(graph, "A:1", "C:5")
    assert pathway is not None
    assert pathway.nodes[0] == "A:1"
    assert pathway.nodes[-1] == "C:5"
    assert 0.0 <= pathway.strength <= 1.0


def test_find_pathway_no_path_returns_none():
    graph = nx.Graph()
    graph.add_node("X")
    graph.add_node("Y")
    assert find_pathway(graph, "X", "Y") is None


def test_rank_pathways():
    graph = build_communication_graph(contact_map=_toy_contact_map())
    pathways = rank_pathways(graph, sources=["A:1", "A:2"], targets=["C:5", "C:6"], top_k=5)
    assert len(pathways) > 0
    assert pathways == sorted(pathways, key=lambda p: p.strength, reverse=True)


def test_betweenness_centrality_keys():
    graph = build_communication_graph(contact_map=_toy_contact_map())
    centrality = betweenness_centrality(graph)
    assert set(centrality.keys()) == set(graph.nodes)


def test_detect_communities_covers_all_nodes():
    graph = build_communication_graph(contact_map=_toy_contact_map())
    communities = detect_communities(graph)
    covered = set().union(*communities) if communities else set()
    assert covered == set(graph.nodes)
    assert -1.0 <= modularity(graph, communities) <= 1.0
