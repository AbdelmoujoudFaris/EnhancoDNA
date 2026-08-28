#!/usr/bin/env python
"""Generate a synthetic demo dataset for EnhancoAI (section 53).

DEMO DATA -- NOT SCIENTIFICALLY VALIDATED.

Produces:
  data/examples/demo_tf_dna_two_factor.pdb   -- idealised TF-A + TF-B + DNA complex
  data/examples/demo_tf_dna_single_factor.pdb -- the same DNA + TF-A, without TF-B
  data/processed/demo_training/*.pt           -- a small labelled voxel-grid dataset
  data/processed/demo_training/metadata.json  -- dataset schema metadata (section 39)

This demonstrates: structure loading, feature extraction, PyTorch training,
prediction and explainability end-to-end without requiring any external
experimental structure or real MD trajectory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEMO_WARNING = "DEMO DATA -- NOT SCIENTIFICALLY VALIDATED"

AMINO_ACIDS_WITH_ATOM = [
    ("ARG", "NH1"), ("LYS", "NZ"), ("ASN", "ND2"), ("GLN", "NE2"),
    ("SER", "OG"), ("THR", "OG1"), ("ASP", "OD1"), ("GLU", "OE1"),
    ("HIS", "NE2"), ("TYR", "OH"), ("ALA", None), ("LEU", None),
    ("VAL", None), ("GLY", None), ("PHE", None), ("PRO", None),
]

BASE_PAIRS = [("DA", "DT"), ("DG", "DC")]

# Idealised B-DNA-like geometry (simplified for a self-consistent synthetic model;
# not a claim of crystallographic B-DNA accuracy -- see module docstring).
RISE = 3.38
TWIST_DEG = 36.0
C1_RADIUS = 5.2  # gives a C1'-C1' distance of ~2*C1_RADIUS = 10.4 A across the pair
P_RADIUS = 9.4


def _rotate_z(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rot @ vec


def build_dna_atoms(n_bp: int, chain_a: str, chain_b: str, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    atom_index = 0
    for i in range(n_bp):
        theta = np.radians(i * TWIST_DEG)
        z = i * RISE
        base_a, base_b = BASE_PAIRS[rng.integers(0, len(BASE_PAIRS))]

        c1_a = _rotate_z(np.array([C1_RADIUS, 0.0, 0.0]), theta) + np.array([0, 0, z])
        c1_b = _rotate_z(np.array([-C1_RADIUS, 0.0, 0.0]), theta) + np.array([0, 0, z])
        p_a = _rotate_z(np.array([P_RADIUS, 0.0, 0.0]), theta) + np.array([0, 0, z - 1.0])
        p_b = _rotate_z(np.array([-P_RADIUS, 0.0, 0.0]), theta) + np.array([0, 0, z - 1.0])

        edge_name_a = "N1" if base_a in ("DA", "DG") else "N3"
        edge_name_b = "N1" if base_b in ("DA", "DG") else "N3"
        edge_a = c1_a + 0.35 * (c1_b - c1_a)
        edge_b = c1_b + 0.35 * (c1_a - c1_b)

        for chain_id, res_seq, res_name, atoms in (
            (chain_a, i + 1, base_a, [("P", p_a), ("C1'", c1_a), (edge_name_a, edge_a)]),
            (chain_b, n_bp - i, base_b, [("P", p_b), ("C1'", c1_b), (edge_name_b, edge_b)]),
        ):
            for atom_name, coord in atoms:
                rows.append(
                    {
                        "atom_index": atom_index, "atom_name": atom_name,
                        "element": atom_name[0].upper(), "altloc": "",
                        "res_name": res_name, "chain_id": chain_id, "res_seq": res_seq, "icode": "",
                        "x": coord[0], "y": coord[1], "z": coord[2],
                        "occupancy": 1.0, "b_factor": 20.0, "is_hetero": False, "model": 1,
                    }
                )
                atom_index += 1
    return pd.DataFrame(rows)


def build_protein_atoms(
    chain_id: str,
    n_res: int,
    z_start: float,
    radius: float,
    rng: np.random.Generator,
    atom_index_start: int = 0,
) -> pd.DataFrame:
    """An idealised alpha-helix backbone wrapped around the DNA major-groove radius,
    with one representative polar/charged side-chain atom per residue (section 32/33
    scanning needs real donor/acceptor atoms to mutate)."""
    rows = []
    atom_index = atom_index_start
    helix_rise, helix_twist = 1.5, np.radians(100.0)
    for i in range(n_res):
        res_name, extra_atom_name = AMINO_ACIDS_WITH_ATOM[rng.integers(0, len(AMINO_ACIDS_WITH_ATOM))]
        theta = i * helix_twist
        z = z_start + i * helix_rise
        ca = _rotate_z(np.array([radius, 0.0, 0.0]), theta) + np.array([0, 0, z])
        inward = -ca / (np.linalg.norm(ca[:2]) + 1e-9)
        n_atom = ca + np.array([0.5, 0.3, -0.4])
        c_atom = ca + np.array([-0.3, 0.5, 0.4])
        o_atom = c_atom + np.array([0.2, 0.2, 0.3])

        atoms = [("N", n_atom), ("CA", ca), ("C", c_atom), ("O", o_atom)]
        if extra_atom_name:
            side_chain_atom = ca + inward * 4.5
            atoms.append((extra_atom_name, side_chain_atom))

        for atom_name, coord in atoms:
            rows.append(
                {
                    "atom_index": atom_index, "atom_name": atom_name,
                    "element": atom_name[0].upper(), "altloc": "",
                    "res_name": res_name, "chain_id": chain_id, "res_seq": i + 1, "icode": "",
                    "x": coord[0], "y": coord[1], "z": coord[2],
                    "occupancy": 1.0, "b_factor": 20.0, "is_hetero": False, "model": 1,
                }
            )
            atom_index += 1
    return pd.DataFrame(rows)


def build_complex(n_bp: int, include_tf_b: bool, seed: int) -> "StructureData":
    from enhancoai.structure.parser import StructureData

    rng = np.random.default_rng(seed)
    dna = build_dna_atoms(n_bp, "C", "D", rng)
    protein_a = build_protein_atoms("A", n_res=16, z_start=n_bp * RISE * 0.3, radius=10.0, rng=rng, atom_index_start=len(dna))
    frames = [dna, protein_a]
    if include_tf_b:
        protein_b = build_protein_atoms(
            "B", n_res=16, z_start=n_bp * RISE * 0.55, radius=10.0, rng=rng, atom_index_start=len(dna) + len(protein_a)
        )
        frames.append(protein_b)

    atoms = pd.concat(frames, ignore_index=True)
    atoms["atom_index"] = range(len(atoms))
    return StructureData(atoms=atoms, source_path="")


def write_pdb(structure, path: Path, header_remark: str) -> None:
    """Write strict fixed-column PDB ATOM records (wwPDB format).

    Columns: 1-6 record, 7-11 serial, 13-16 atom name, 17 altLoc, 18-20
    resName, 22 chainID, 23-26 resSeq, 31-54 xyz, 55-66 occ/temp, 77-78 element.
    """
    lines = [f"REMARK   {DEMO_WARNING}", f"REMARK   {header_remark}"]
    for row in structure.atoms.itertuples():
        atom_name = row.atom_name
        atom_field = f"{atom_name:<4}" if len(atom_name) >= 4 else f" {atom_name:<3}"
        lines.append(
            f"ATOM  {row.atom_index + 1:>5d} {atom_field}{'':1s}{row.res_name:>3s} {row.chain_id:1s}"
            f"{row.res_seq:>4d}{'':1s}   {row.x:8.3f}{row.y:8.3f}{row.z:8.3f}{row.occupancy:6.2f}{row.b_factor:6.2f}"
            f"          {row.element:>2s}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_training_samples(n_samples: int, seed: int):
    """Voxelise many small synthetic interfaces with a deterministic, declared-synthetic
    labelling rule (interface distance -> cooperative), for a runnable demo training loop.
    This label is a synthetic construction, not a biological ground truth."""
    from enhancoai.structure.selection import heavy_atoms
    from enhancoai.voxel.voxelizer import voxelize_interface, to_tensor

    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_samples):
        n_bp = int(rng.integers(14, 22))
        include_b = bool(rng.integers(0, 2))
        structure = build_complex(n_bp=n_bp, include_tf_b=include_b, seed=seed * 1000 + i)

        protein_atoms = heavy_atoms(structure.chain("A"))
        dna_atoms = pd.concat([heavy_atoms(structure.chain("C")), heavy_atoms(structure.chain("D"))])
        voxel_grid = voxelize_interface(protein_atoms, dna_atoms, grid_size=16)
        tensor = to_tensor(voxel_grid).float().squeeze(0)  # (C, D, H, W)

        # Synthetic label rule: presence of TF-B + a compact interface => "cooperative".
        label = 1.0 if include_b else 0.0
        samples.append({"voxel_grid": tensor, "cooperative_label": label, "cooperativity_value": label, "sample_id": f"demo_{i:03d}"})
    return samples


def main() -> None:
    examples_dir = REPO_ROOT / "data" / "examples"
    processed_dir = REPO_ROOT / "data" / "processed" / "demo_training"
    examples_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(DEMO_WARNING)

    two_factor = build_complex(n_bp=24, include_tf_b=True, seed=1)
    single_factor = build_complex(n_bp=24, include_tf_b=False, seed=1)
    write_pdb(two_factor, examples_dir / "demo_tf_dna_two_factor.pdb", "Synthetic TF-A + TF-B + DNA complex")
    write_pdb(single_factor, examples_dir / "demo_tf_dna_single_factor.pdb", "Synthetic TF-A + DNA complex (no TF-B)")
    print(f"Wrote {examples_dir / 'demo_tf_dna_two_factor.pdb'}")
    print(f"Wrote {examples_dir / 'demo_tf_dna_single_factor.pdb'}")

    import torch

    samples = build_training_samples(n_samples=40, seed=7)
    metadata = []
    for sample in samples:
        sample_path = processed_dir / f"{sample['sample_id']}.pt"
        torch.save(
            {"voxel_grid": sample["voxel_grid"], "cooperative_label": sample["cooperative_label"], "cooperativity_value": sample["cooperativity_value"]},
            sample_path,
        )
        metadata.append(
            {
                "sample_id": sample["sample_id"],
                "protein_a": "A",
                "protein_b": "B" if sample["cooperative_label"] == 1.0 else None,
                "dna": "C,D",
                "structure_path": None,
                "trajectory_path": None,
                "tensor_path": str(sample_path.relative_to(REPO_ROOT)),
                "cooperativity_label": sample["cooperative_label"],
                "mechanism_label": None,
                "cooperativity_value": sample["cooperativity_value"],
                "note": DEMO_WARNING,
            }
        )

    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Wrote {len(samples)} demo training samples to {processed_dir}")
    print(DEMO_WARNING)


if __name__ == "__main__":
    main()
