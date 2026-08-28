# Explainable AI and the Digital Perturbation Laboratory

All methods in `enhancoai.explainability` are implemented directly on
`torch.autograd` -- no extra dependency (e.g. Captum) is required.

- `saliency.saliency_map` -- |d output / d input|.
- `integrated_gradients.integrated_gradients` -- path integral from a
  baseline (default all-zeros) to the input (Sundararajan et al. 2017).
- `gradcam.grad_cam_3d` -- Grad-CAM on the 3D CNN's last residual block,
  via forward/backward hooks.
- `attention.cross_modal_attention_weights` -- the hybrid model's
  modality-to-modality attention matrix.
- `attention.gnn_edge_importance` -- gradient-based edge-importance proxy
  for the pure-PyTorch GNN.
- `mapping.node_scores_to_frame` / `ranked_entities` -- turn a per-node
  score tensor into a `(chain_id, res_seq, importance)` table and a
  ranked "Rank | Entity | Position | Importance" report, for mapping onto
  the 3D structure (`enhancoai.visualization.structure.render_attribution_overlay`).

Always report the full interpretation (section 61), never a bare number:

```python
prediction = predictor.predict(voxel_grid=voxel_tensor)
prediction.to_dict()
# {"is_cooperative": ..., "probability": ..., "mechanism": ...,
#  "confidence": ..., "evidence": [...], "weights_available": ...}
```

If `weights_available` is `False`, every other field is a placeholder and
`evidence` explains why -- never a fabricated prediction.

## Digital Perturbation Laboratory

`enhancoai.perturbation` runs the **AI model** on a mutated input
representation (a residue's one-hot identity swapped to ALA, or a DNA
base swapped to another canonical base) and reports the predicted-
cooperativity delta:

```python
from enhancoai.perturbation.residue_scan import alanine_scan
from enhancoai.perturbation.dna_scan import dna_base_scan
from enhancoai.perturbation.interaction_map import build_interaction_map

residue_deltas = alanine_scan(predictor, graph_x, graph_edge_index, node_ids, chain_id="A")
dna_deltas = dna_base_scan(predictor, graph_x, graph_edge_index, node_ids, chain_id="C", original_bases=bases)
interaction_map = build_interaction_map(residue_deltas, dna_deltas)
```

Every result row carries `method = "AI-based in-silico perturbation"` --
this is a prediction under a perturbed representation, not a re-run
molecular simulation. `build_interaction_map` sums two *independent*
single-mutation effects as a screening heuristic to flag candidate
residue/base pairs for joint follow-up, not a jointly re-evaluated double
mutant.
