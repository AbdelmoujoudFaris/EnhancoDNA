#!/usr/bin/env python
"""RMSD/RMSF/correlation analysis over an MD trajectory (or a single-frame structure).

Example:
    python scripts/analyse_md.py --topology data/examples/demo_tf_dna_two_factor.pdb \\
        --output-dir experiments/md_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--protein-selection", default="protein and name CA")
    parser.add_argument("--dna-selection", default="nucleic")
    parser.add_argument("--output-dir", default="experiments/md")
    args = parser.parse_args()

    from enhancoai.md.loader import load_trajectory
    from enhancoai.md.rmsd import compute_rmsd
    from enhancoai.md.rmsf import compute_rmsf
    from enhancoai.md.correlations import dynamic_cross_correlation
    from enhancoai.visualization.plots import plot_timeseries, plot_rmsf
    from enhancoai.visualization.contacts import render_correlation_matrix

    handle = load_trajectory(args.topology, args.trajectory)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {handle.n_frames} frame(s).")

    for label, selection in (("protein", args.protein_selection), ("dna", args.dna_selection)):
        try:
            rmsd = compute_rmsd(handle, selection=selection)
            rmsf = compute_rmsf(handle, selection=selection)
        except ValueError as exc:
            print(f"{label}: {exc}")
            continue
        rmsd.to_csv(output_dir / f"rmsd_{label}.csv", index=False)
        rmsf.to_csv(output_dir / f"rmsf_{label}.csv", index=False)
        plot_timeseries(rmsd, "time_ps", "rmsd", title=f"{label} RMSD").savefig(output_dir / f"rmsd_{label}.png", dpi=120)
        plot_rmsf(rmsf, title=f"{label} RMSF").savefig(output_dir / f"rmsf_{label}.png", dpi=120)
        print(f"{label}: RMSD mean={rmsd['rmsd'].mean():.3f} A, RMSF mean={rmsf['rmsf'].mean():.3f} A")

    if handle.n_frames > 1:
        dccm = dynamic_cross_correlation(handle, selection=args.protein_selection)
        dccm.to_csv(output_dir / "dccm_protein.csv")
        render_correlation_matrix(dccm).savefig(output_dir / "dccm_protein.png", dpi=120)
    else:
        print("Single-frame input: skipping dynamic cross-correlation (requires >1 frame).")

    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
