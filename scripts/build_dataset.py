#!/usr/bin/env python
"""Build a voxel-grid training dataset from a manifest of structures (sections 39-40).

Manifest CSV columns: sample_id, structure_path, protein_a, protein_b (optional,
blank if none), dna_chains (comma-separated), cooperativity_label (0/1),
cooperativity_value (float, optional), mechanism_label (optional).

Splits are clustered by approximate protein-A sequence identity (via
difflib.SequenceMatcher, NOT a rigorous alignment -- see docs) so that
near-duplicate complexes never straddle train/val/test, avoiding the
structural leakage warned about in section 40.

Example:
    python scripts/build_dataset.py --manifest manifest.csv --output-dir data/processed/my_dataset
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def chain_sequence(structure, chain_id: str) -> str:
    residues = structure.chain(chain_id).drop_duplicates(subset=["res_seq"]).sort_values("res_seq")
    return "".join(THREE_TO_ONE.get(r.strip().upper(), "X") for r in residues["res_name"])


def cluster_by_identity(sequences: dict[str, str], identity_cutoff: float) -> dict[str, int]:
    """Greedy single-linkage clustering: assign each sample to the first existing
    cluster whose representative sequence is >= identity_cutoff similar."""
    cluster_reps: list[str] = []
    assignment: dict[str, int] = {}
    for sample_id, seq in sequences.items():
        placed = False
        for cluster_id, rep in enumerate(cluster_reps):
            ratio = difflib.SequenceMatcher(None, seq, rep).ratio()
            if ratio >= identity_cutoff:
                assignment[sample_id] = cluster_id
                placed = True
                break
        if not placed:
            cluster_reps.append(seq)
            assignment[sample_id] = len(cluster_reps) - 1
    return assignment


def split_by_cluster(assignment: dict[str, int], val_fraction: float, test_fraction: float, seed: int) -> dict[str, str]:
    import random

    rng = random.Random(seed)
    clusters = sorted(set(assignment.values()))
    rng.shuffle(clusters)

    n_val = max(1, int(round(len(clusters) * val_fraction))) if clusters else 0
    n_test = max(1, int(round(len(clusters) * test_fraction))) if clusters else 0
    val_clusters = set(clusters[:n_val])
    test_clusters = set(clusters[n_val : n_val + n_test])

    split = {}
    for sample_id, cluster_id in assignment.items():
        if cluster_id in val_clusters:
            split[sample_id] = "val"
        elif cluster_id in test_clusters:
            split[sample_id] = "test"
        else:
            split[sample_id] = "train"
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--identity-cutoff", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd
    import torch

    from enhancoai.structure.parser import load_structure
    from enhancoai.structure.selection import heavy_atoms
    from enhancoai.voxel.voxelizer import voxelize_interface, to_tensor

    manifest = pd.read_csv(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences, metadata = {}, []
    for row in manifest.itertuples():
        structure = load_structure(row.structure_path)
        dna_chains = [c.strip() for c in str(row.dna_chains).split(",")]

        protein_atoms = heavy_atoms(structure.chain(row.protein_a))
        dna_atoms = pd.concat([heavy_atoms(structure.chain(c)) for c in dna_chains])
        voxel_grid = voxelize_interface(protein_atoms, dna_atoms, grid_size=args.grid_size)
        tensor = to_tensor(voxel_grid).float().squeeze(0)

        tensor_path = output_dir / f"{row.sample_id}.pt"
        payload = {"voxel_grid": tensor}
        if hasattr(row, "cooperativity_label") and pd.notna(row.cooperativity_label):
            payload["cooperative_label"] = float(row.cooperativity_label)
        if hasattr(row, "cooperativity_value") and pd.notna(row.cooperativity_value):
            payload["cooperativity_value"] = float(row.cooperativity_value)
        torch.save(payload, tensor_path)

        sequences[row.sample_id] = chain_sequence(structure, row.protein_a)
        metadata.append(
            {
                "sample_id": row.sample_id,
                "protein_a": row.protein_a,
                "protein_b": getattr(row, "protein_b", None),
                "dna": ",".join(dna_chains),
                "structure_path": row.structure_path,
                "tensor_path": str(tensor_path.relative_to(output_dir.parent) if output_dir.parent in tensor_path.parents else tensor_path),
                "cooperativity_label": payload.get("cooperative_label"),
                "mechanism_label": getattr(row, "mechanism_label", None),
                "cooperativity_value": payload.get("cooperativity_value"),
            }
        )
        print(f"Processed {row.sample_id}")

    assignment = cluster_by_identity(sequences, args.identity_cutoff)
    split = split_by_cluster(assignment, args.val_fraction, args.test_fraction, args.seed)
    for entry in metadata:
        entry["split"] = split[entry["sample_id"]]
        entry["sequence_cluster"] = assignment[entry["sample_id"]]

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    n_train = sum(1 for v in split.values() if v == "train")
    n_val = sum(1 for v in split.values() if v == "val")
    n_test = sum(1 for v in split.values() if v == "test")
    print(f"\nSplit (clustered at {args.identity_cutoff:.0%} approximate identity): "
          f"train={n_train}, val={n_val}, test={n_test}, n_clusters={len(set(assignment.values()))}")
    print(f"Dataset written to {output_dir}")


if __name__ == "__main__":
    main()
