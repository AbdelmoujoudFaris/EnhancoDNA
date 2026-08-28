# Cooperativity Metrics

## PMFs

```python
from enhancoai.free_energy.pmf import compute_pmf_1d

pmf = compute_pmf_1d(samples, n_bins=50, temperature_k=300.0)
```

`ΔG(x) = -kB T ln P(x) + C`, computed from the observed histogram of an
**equilibrium** reaction-coordinate sample. This is only a valid
free-energy estimate if the input truly is unbiased/equilibrium sampling;
bins with too few samples are reported as `NaN`, not extrapolated, and a
`warnings` list flags small-sample runs. This is *not* umbrella
sampling/WHAM/metadynamics reweighting -- do not present it as such.

## Binding free energy from a PMF

```python
from enhancoai.free_energy.cooperativity import binding_free_energy

g_bind = binding_free_energy(pmf, bound_state="low")  # "low" if small x = bound (e.g. a distance)
```

Computed as the count-weighted mean free energy at the bound end of the
coordinate minus the mean at the unbound end -- *not* simply the PMF's
own minimum (which is always ~0 after normalisation and would make any
two-system comparison vacuous).

## ΔΔG_coop

```python
from enhancoai.free_energy.cooperativity import cooperativity_free_energy

result = cooperativity_free_energy(pmf_a_alone, pmf_a_with_b, bound_state="low")
result.delta_delta_g_coop  # G(A with B) - G(A alone); negative = TF-B stabilises TF-A binding
```

Sign convention: `delta_delta_g_coop = ΔG(A binding | B present) - ΔG(A
binding alone)`. Negative means cooperative stabilisation, positive means
antagonism. This is a simulation-derived proxy, not an experimental
thermodynamic measurement -- see `result.caveats`.

## Cooperativity Score (section 22)

```python
from enhancoai.features.interaction import CooperativityEvidence, evidence_to_components
from enhancoai.free_energy.cooperativity import compute_cooperativity_score, CooperativityScoreWeights

evidence = CooperativityEvidence(
    energetic_shift=1.2, dynamic_shift=0.4, orientation_shift=15.0,
    dna_pathway_strength=0.6, interface_persistence=0.7,
)
components = evidence_to_components(evidence)
result = compute_cooperativity_score(components, CooperativityScoreWeights())
result.total_score, result.components  # transparent, decomposable
```

Weights are configurable (`configs/cooperativity.yaml`) and always
renormalised to sum to 1. This is a heuristic composite indicator for
comparing complexes within the same analysis, not a validated biological
constant -- never present it as an absolute probability of cooperativity.
