"""Build the protein-DNA interaction graph consumed by the GNN model.

Nodes: protein residues (represented by CA) and DNA nucleotides
(represented by C1'). Edges: covalent (sequential residues/nucleotides in
the same chain), spatial contact (distance cutoff), and optionally dynamic
correlation (if a DCCM/MI matrix is supplied). Uses PyTorch Geometric's
``Data`` container when available, otherwise falls back to a minimal
dataclass with the same field names (``x``, ``edge_index``, ``edge_attr``)
so :mod:`enhancoai.models.gnn` works either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from enhancoai.graph.features import node_feature_vector, NODE_FEATURE_DIM

EDGE_TYPE_COVALENT = 0
EDGE_TYPE_CONTACT = 1
EDGE_TYPE_HYDROGEN_BOND = 2
EDGE_TYPE_CORRELATION = 3


@dataclass
class GraphData:
    x: "object"  # torch.Tensor (n_nodes, NODE_FEATURE_DIM)
    edge_index: "object"  # torch.Tensor (2, n_edges)
    edge_attr: "object"  # torch.Tensor (n_edges, edge_feature_dim)
    node_ids: list[str] = field(default_factory=list)


def _representative_atoms(structure, chain_id: str, is_protein: bool) -> pd.DataFrame:
    chain = structure.chain(chain_id)
    atom_name = "CA" if is_protein else "C1'"
    rep = chain[chain["atom_name"].str.strip() == atom_name]
    if rep.empty:
        # fall back to first atom of each residue if the representative atom is missing
        rep = chain.drop_duplicates(subset=["res_seq"])
    return rep.drop_duplicates(subset=["res_seq"]).sort_values("res_seq")


def build_protein_dna_graph(
    structure,
    protein_chains: list[str],
    dna_chains: list[str],
    contact_cutoff: float = 8.0,
    correlation_matrix: pd.DataFrame | None = None,
    correlation_chain_id: str | None = None,
) -> GraphData:
    import torch

    node_rows = []
    node_ids: list[str] = []
    chain_type: dict[str, bool] = {}

    for chain_id in protein_chains:
        chain_type[chain_id] = True
        rep = _representative_atoms(structure, chain_id, is_protein=True)
        for row in rep.itertuples():
            node_ids.append(f"{chain_id}:{row.res_seq}")
            node_rows.append((chain_id, row.res_seq, row.res_name, True, row.x, row.y, row.z))

    for chain_id in dna_chains:
        chain_type[chain_id] = False
        rep = _representative_atoms(structure, chain_id, is_protein=False)
        for row in rep.itertuples():
            node_ids.append(f"{chain_id}:{row.res_seq}")
            node_rows.append((chain_id, row.res_seq, row.res_name, False, row.x, row.y, row.z))

    if not node_rows:
        raise ValueError("No nodes could be built: check protein_chains/dna_chains selection.")

    features = np.stack([node_feature_vector(r[2], r[3]) for r in node_rows])
    coords = np.array([[r[4], r[5], r[6]] for r in node_rows])

    edges: list[tuple[int, int]] = []
    edge_types: list[int] = []
    edge_distances: list[float] = []

    # covalent edges: sequential residues within the same chain
    index_by_id = {node_id: i for i, node_id in enumerate(node_ids)}
    chains_seen: dict[str, list[int]] = {}
    for i, (chain_id, res_seq, *_rest) in enumerate(node_rows):
        chains_seen.setdefault(chain_id, []).append(i)
    for chain_id, indices in chains_seen.items():
        ordered = sorted(indices, key=lambda i: node_rows[i][1])
        for a, b in zip(ordered[:-1], ordered[1:]):
            edges.append((a, b))
            edge_types.append(EDGE_TYPE_COVALENT)
            edge_distances.append(float(np.linalg.norm(coords[a] - coords[b])))

    # spatial contact edges
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=contact_cutoff)
    for a, b in pairs:
        edges.append((a, b))
        edge_types.append(EDGE_TYPE_CONTACT)
        edge_distances.append(float(np.linalg.norm(coords[a] - coords[b])))

    # dynamic correlation edges (optional, single-chain resid-indexed matrix)
    if correlation_matrix is not None and correlation_chain_id is not None:
        ids = correlation_matrix.index.tolist()
        for i, res_i in enumerate(ids):
            key_i = f"{correlation_chain_id}:{res_i}"
            if key_i not in index_by_id:
                continue
            for res_j in ids[i + 1 :]:
                key_j = f"{correlation_chain_id}:{res_j}"
                if key_j not in index_by_id:
                    continue
                corr = correlation_matrix.loc[res_i, res_j]
                if abs(corr) < 0.3:
                    continue
                a, b = index_by_id[key_i], index_by_id[key_j]
                edges.append((a, b))
                edge_types.append(EDGE_TYPE_CORRELATION)
                edge_distances.append(float(np.linalg.norm(coords[a] - coords[b])))

    if not edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 2), dtype=np.float32)
    else:
        undirected = edges + [(b, a) for a, b in edges]
        undirected_types = edge_types + edge_types
        undirected_dist = edge_distances + edge_distances
        edge_index = np.array(undirected, dtype=np.int64).T
        edge_attr = np.stack(
            [np.array(undirected_types, dtype=np.float32), np.array(undirected_dist, dtype=np.float32)],
            axis=1,
        )

    x = torch.from_numpy(features.astype(np.float32))
    edge_index_t = torch.from_numpy(edge_index)
    edge_attr_t = torch.from_numpy(edge_attr)

    try:
        from torch_geometric.data import Data

        return Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t, node_ids=node_ids)
    except ImportError:
        return GraphData(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t, node_ids=node_ids)
