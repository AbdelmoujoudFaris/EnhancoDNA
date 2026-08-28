# DNA Geometry and Allosteric Networks

## DNA geometry (simplified, documented approximations)

`enhancoai.dna` does **not** reimplement 3DNA/Curves+. It provides:

- `geometry.find_watson_crick_pairs` -- geometric pairing (C1'-C1'
  distance in [9, 11.5] A plus Watson-Crick-edge-atom proximity), not
  hydrogen-bond-based pairing.
- `geometry.base_pair_frames` -- a local reference frame per base pair
  (origin = C1'-C1' midpoint; z-axis = locally estimated helical tangent;
  x-axis = strand-b-to-strand-a C1' direction, Gram-Schmidt orthogonalised).
- `helical_parameters.step_parameters` -- shift/slide/rise/tilt/roll/twist
  between consecutive base-pair frames, via the standard mid-frame
  convention, but a simplified reimplementation.
- `curvature.local_curvature` / `global_bend_angle` -- bending angle from
  the sequence of base-pair origins.
- `groove_analysis.groove_widths` -- P-P distance minus a fixed vdW
  correction (El Hassan & Calladine-style convention); a geometric
  approximation for trend analysis, not dedicated groove software.

Use these for comparative/trend analysis (e.g. "does DNA curvature
increase when TF-B is added?"), not as a substitute for a validated
groove-geometry tool when precise absolute values matter.

## Allosteric communication network

```python
from enhancoai.allostery.network import build_communication_graph
from enhancoai.allostery.pathways import rank_pathways, betweenness_centrality
from enhancoai.allostery.communities import detect_communities

graph = build_communication_graph(
    contact_map=residue_contact_map,          # from enhancoai.interactions
    correlation_matrix=dccm, correlation_chain_id="A",
    mutual_information_matrix=mi, mi_chain_id="A",
)
pathways = rank_pathways(graph, sources=[...], targets=[...], top_k=10)
```

Nodes are `"{chain_id}:{res_seq}"` strings covering both protein residues
and DNA nucleotides. Edges combine any subset of: physical contact
(weight ~ 1/distance), dynamic cross-correlation (|correlation| above a
threshold), and mutual information -- each edge records which evidence
type(s) contributed (`graph[u][v]["evidence"]`), so a pathway's provenance
is always traceable.

Pathway "strength" is a bottleneck (weakest-edge) measure along the
shortest path, normalised to [0, 1] against the graph's strongest edge --
a candidate-communication-pathway score, not a rigorous free-energy or
kinetic rate.

**Caveat**: correlation and mutual information are statistical
associations. They are evidence for, not proof of, a causal allosteric
pathway -- corroborate with the free-energy/cooperativity-score evidence
before drawing a mechanistic conclusion (see the GUI's Hypothesis Testing
tab, which enforces this).
