#!/usr/bin/env python
"""Run cooperativity inference on a structure using a trained (or untrained) checkpoint.

Example:
    python scripts/predict.py --model experiments/demo_run/checkpoint.pt \\
        --input data/examples/demo_tf_dna_two_factor.pdb --protein A --dna C --dna D
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a .pt checkpoint (need not exist).")
    parser.add_argument("--input", required=True, help="Structure file (PDB/mmCIF).")
    parser.add_argument("--protein", required=True)
    parser.add_argument("--dna", action="append", required=True, dest="dna_chains")
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--embedding-dim", type=int, default=512, help="Must match the checkpoint's training config (model.embedding_dim).")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    import pandas as pd

    from enhancoai.structure.parser import load_structure
    from enhancoai.structure.selection import heavy_atoms
    from enhancoai.voxel.voxelizer import voxelize_interface, to_tensor
    from enhancoai.inference.predictor import Predictor
    from enhancoai.utils.config import ModelConfig

    structure = load_structure(args.input)
    protein_atoms = heavy_atoms(structure.chain(args.protein))
    dna_atoms = pd.concat([heavy_atoms(structure.chain(c)) for c in args.dna_chains])
    voxel_grid = voxelize_interface(protein_atoms, dna_atoms, grid_size=args.grid_size)
    voxel_tensor = to_tensor(voxel_grid).float()

    predictor = Predictor(args.model, ModelConfig(architecture="cnn3d", embedding_dim=args.embedding_dim))
    prediction = predictor.predict(voxel_grid=voxel_tensor)

    result = prediction.to_dict()
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
