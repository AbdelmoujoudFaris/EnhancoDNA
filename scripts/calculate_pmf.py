#!/usr/bin/env python
"""Compute a 1D PMF from a CSV column of reaction-coordinate samples.

Example:
    python scripts/calculate_pmf.py --input samples.csv --column com_distance --temperature 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV file with reaction-coordinate samples.")
    parser.add_argument("--column", required=True)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--output-dir", default="experiments/pmf")
    args = parser.parse_args()

    import pandas as pd

    from enhancoai.free_energy.pmf import compute_pmf_1d, pmf_to_frame
    from enhancoai.visualization.plots import plot_pmf

    samples = pd.read_csv(args.input)[args.column].to_numpy()
    pmf = compute_pmf_1d(samples, n_bins=args.n_bins, temperature_k=args.temperature)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pmf_to_frame(pmf).to_csv(output_dir / "pmf.csv", index=False)
    plot_pmf(pmf).savefig(output_dir / "pmf.png", dpi=120)

    print(f"Method: {pmf.method}")
    print(f"n_samples: {pmf.n_samples}")
    for warning in pmf.warnings:
        print(f"WARNING: {warning}")
    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
