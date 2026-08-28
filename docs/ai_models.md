# AI Models

Four real PyTorch architectures, all in `enhancoai.models`, selected via
`ModelConfig.architecture` (`configs/*.yaml` -> `model.architecture`).

| Architecture | Module | Input | Notes |
|---|---|---|---|
| 3D CNN | `cnn3d.py` | 10-channel voxel grid (`enhancoai.voxel`) | Conv3D + residual blocks + global pool |
| GNN | `gnn.py` | protein-residue/DNA-nucleotide graph (`enhancoai.graph`) | Uses PyTorch Geometric if installed, else a pure-PyTorch scatter-add graph conv |
| Temporal | `temporal.py` | (batch, n_frames, 4) per-frame MD features (`enhancoai.md.trajectory_features`) | Transformer encoder |
| Hybrid | `hybrid.py` | any subset of the above | Cross-modal attention fusion, 5 task heads, missing modalities replaced by a learned placeholder |

## Voxel channels (3D CNN input)

`protein_occupancy, dna_occupancy, electrostatic_potential,
hydrophobicity, hydrogen_bond_donor, hydrogen_bond_acceptor, dna_backbone,
base_identity, interface_density, distance_field` -- see
`enhancoai.voxel.representation.CHANNEL_NAMES`. Atoms are splatted with a
Gaussian kernel, not hard binary occupancy.

## Graph nodes/edges

Nodes: protein residues (CA-represented) and DNA nucleotides
(C1'-represented), 28-dim feature vector (one-hot identity + is_protein/
is_dna + hydrophobicity + charge). Edges: covalent (sequential in-chain),
spatial contact (KD-tree cutoff), optional dynamic-correlation edges.

## Multi-task heads (hybrid model)

`cooperativity_logit` (binary), `cooperativity_strength` (regression, in
[0, 1]), `mechanism_logits` (4-way: direct / DNA-mediated / mixed / weak-
or-none, configurable), `residue_importance` and `dna_importance`
(per-node regression). Loss weights are configurable
(`ModelConfig.loss_weights` / `LossWeights`); a task is skipped
automatically if its label is absent from a batch.

## Training

`enhancoai.training.trainer.Trainer` is model-agnostic: it forwards
whichever of `voxel_grid`/`graph_x`/`graph_edge_index`/`graph_batch`/
`frame_features` are present in a batch dict. Supports pause/resume/stop
via `Trainer.control` (used by the GUI's training controls) and writes the
standard experiment directory (`config.yaml`, `metrics.json`,
`history.csv`, `checkpoint.pt`, `figures/`).

## GPU

`enhancoai.utils.device.detect_device()` picks CUDA > MPS > CPU
automatically (never assumes a specific GPU) and reports what it found;
pass `prefer="cpu"` to force CPU.
