# Changelog

All notable changes to EnhancoAI are documented in this file.

## [Unreleased]

### Added

- Initial public scaffold: structure loading/cleaning/chain classification,
  protein-DNA and protein-protein interaction analysis, simplified DNA
  helical-parameter/curvature/groove geometry, MD trajectory analysis
  (RMSD/RMSF/contacts/correlations/orientation), PMF and cooperativity
  free-energy utilities, DNA-mediated allosteric network construction
  (contacts + DCCM + mutual information, pathway ranking, community
  detection), 3D-CNN/GNN/temporal/hybrid PyTorch models, multi-task
  training loop with pause/resume/stop, explainability (saliency,
  integrated gradients, Grad-CAM, GNN edge importance), in-silico
  mutational scanning, PyVista-optional 3D + matplotlib 2D visualisation,
  HTML/PDF report generation, a PySide6 desktop GUI, and a `enhancoai` CLI.
- Synthetic demo dataset generator (`scripts/generate_demo_dataset.py`),
  clearly labelled DEMO DATA -- NOT SCIENTIFICALLY VALIDATED.

## [0.1.0] - 2026-08-28

- Project scaffold created.
