# MD Trajectory Analysis

Backed by MDAnalysis, so any format it reads works
(`.xtc`, `.trr`, `.dcd`, `.nc`, `.pdb`, `.gro`, ...).

```python
from enhancoai.md.loader import load_trajectory
from enhancoai.md.rmsd import compute_rmsd
from enhancoai.md.rmsf import compute_rmsf
from enhancoai.md.correlations import dynamic_cross_correlation, mutual_information_matrix
from enhancoai.md.orientation import orientation_trajectory
from enhancoai.md.contacts import contact_persistence, center_of_mass_distance

handle = load_trajectory("system.pdb", "trajectory.xtc", stride=1)
rmsd = compute_rmsd(handle, selection="protein and name CA")
rmsf = compute_rmsf(handle, selection="protein and name CA")
```

A topology file alone (no trajectory) is treated as a single-frame
trajectory -- useful for smoke-testing the pipeline against a static
structure, though RMSD/RMSF are then trivially zero.

## Superposition

RMSD/RMSF/DCCM all superpose frames via the Kabsch algorithm to the first
frame (RMSD) or to the mean structure (RMSF/DCCM) before computing
fluctuations -- translation and rotation are removed first, as is
standard.

## Correlation vs mutual information

`dynamic_cross_correlation` captures *linear* correlation of residue
displacement vectors (directional). `mutual_information_matrix` uses a
coarse histogram estimator on displacement magnitude (nonlinear, but
direction-blind and needs many frames per bin to be reliable -- see
[`reproducibility.md`](reproducibility.md) for sample-size caveats).

## Orientation

`orientation_trajectory` tracks the angle between the first principal axis
of two atom selections (e.g. a TF chain vs the DNA, or TF-A vs TF-B) frame
by frame -- the basis for testing "does TF-B change TF-A's orientation?"
(Q1 in `scientific_background.md`).
