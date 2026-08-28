"""Allosteric communication network and pathway visualisation (sections 20, 47)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def render_network(graph: nx.Graph, highlight_nodes: set[str] | None = None, title: str = "Communication network"):
    fig, ax = plt.subplots(figsize=(7, 7))
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
        ax.axis("off")
        return fig

    layout = nx.spring_layout(graph, weight="weight", seed=42)
    weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    max_weight = max(weights) if weights else 1.0
    widths = [1 + 3 * (w / max_weight) for w in weights]

    node_colors = ["#d62728" if highlight_nodes and n in highlight_nodes else "#1f77b4" for n in graph.nodes()]

    nx.draw_networkx_nodes(graph, layout, node_size=60, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(graph, layout, width=widths, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(graph, layout, font_size=6, ax=ax)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_pathway_ranking(pathways, title: str = "Top communication pathways"):
    fig, ax = plt.subplots(figsize=(6, max(2, 0.4 * len(pathways))))
    if not pathways:
        ax.text(0.5, 0.5, "No pathways found", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = [" -> ".join(p.nodes) for p in pathways]
    strengths = [p.strength for p in pathways]
    ax.barh(range(len(pathways)), strengths, color="#2ca02c")
    ax.set_yticks(range(len(pathways)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Pathway strength")
    ax.set_title(title)
    fig.tight_layout()
    return fig
