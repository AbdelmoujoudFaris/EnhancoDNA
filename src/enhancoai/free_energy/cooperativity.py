"""Cooperativity free energy and the composite Cooperativity Score (sections 18, 22).

Sign convention for the thermodynamic coupling free energy:

    delta_delta_g_coop = delta_g(A binding | B present) - delta_g(A binding alone)

A negative value means TF-B binding *stabilises* TF-A binding (positive
cooperativity, in the usual thermodynamic-cycle convention where more
negative ΔG is more favourable). A positive value means TF-B binding is
antagonistic. This is a simulation-derived proxy (estimated from relative
population/PMF-minimum shifts between the two systems), not a claim of
experimental thermodynamic equivalence -- see section 18 and the
limitations documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from enhancoai.free_energy.pmf import PMFResult, BOLTZMANN_KCAL_PER_MOL_K, free_energy_at_extremes


@dataclass
class CooperativityFreeEnergy:
    delta_g_a_alone: float
    delta_g_a_with_b: float
    delta_delta_g_coop: float
    temperature_k: float
    method: str
    caveats: list[str] = field(default_factory=list)


def binding_free_energy(pmf: PMFResult, bound_state: str = "low") -> float:
    """Estimate a binding free energy from a PMF's bound-end vs unbound-end values.

    ``bound_state`` says which end of the reaction coordinate is the bound
    state: "low" for a coordinate where small values mean bound (e.g. a
    protein-DNA centre-of-mass distance), "high" for one where large values
    mean bound (e.g. a native-contact fraction or contact count). The
    result is G(bound end) - G(unbound end); a negative value means the
    bound state is more probable/favourable in the sampled ensemble.
    """
    if bound_state not in ("low", "high"):
        raise ValueError("bound_state must be 'low' or 'high'.")
    mean_low, mean_high = free_energy_at_extremes(pmf)
    if np.isnan(mean_low) or np.isnan(mean_high):
        raise ValueError("PMF has insufficient sampling at one or both coordinate extremes.")
    return (mean_low - mean_high) if bound_state == "low" else (mean_high - mean_low)


def cooperativity_free_energy(
    pmf_a_alone: PMFResult,
    pmf_a_with_b: PMFResult,
    bound_state: str = "low",
) -> CooperativityFreeEnergy:
    """Compute ΔΔG_coop from two PMFs of the same reaction coordinate.

    ``pmf_a_alone``: PMF of TF-A binding to DNA in the TF-A + DNA system.
    ``pmf_a_with_b``: PMF of the same coordinate in the TF-A + TF-B + DNA system.
    ``bound_state``: which end of the coordinate is "bound" -- see
    :func:`binding_free_energy`. Both PMFs must use the same convention.
    """
    if pmf_a_alone.temperature_k != pmf_a_with_b.temperature_k:
        raise ValueError("PMFs must be computed at the same temperature to compare.")

    g_alone = binding_free_energy(pmf_a_alone, bound_state)
    g_with_b = binding_free_energy(pmf_a_with_b, bound_state)
    ddg = g_with_b - g_alone

    caveats = [
        "Derived from simulation PMF bound-vs-unbound-end differences, not experimental binding assays.",
        "Assumes both PMFs are computed over the same reaction coordinate definition, bound/unbound "
        "convention, and comparable sampling.",
    ]
    caveats.extend(pmf_a_alone.warnings)
    caveats.extend(pmf_a_with_b.warnings)

    return CooperativityFreeEnergy(
        delta_g_a_alone=g_alone,
        delta_g_a_with_b=g_with_b,
        delta_delta_g_coop=ddg,
        temperature_k=pmf_a_alone.temperature_k,
        method="PMF bound/unbound-end difference (equilibrium histogram based)",
        caveats=caveats,
    )


@dataclass
class CooperativityScoreWeights:
    energetic_coupling: float = 0.25
    dynamic_coupling: float = 0.2
    orientation_coupling: float = 0.2
    dna_communication: float = 0.2
    interface_persistence: float = 0.15

    def normalised(self) -> "CooperativityScoreWeights":
        total = (
            self.energetic_coupling
            + self.dynamic_coupling
            + self.orientation_coupling
            + self.dna_communication
            + self.interface_persistence
        )
        if total <= 0:
            raise ValueError("Cooperativity score weights must sum to a positive value.")
        return CooperativityScoreWeights(
            energetic_coupling=self.energetic_coupling / total,
            dynamic_coupling=self.dynamic_coupling / total,
            orientation_coupling=self.orientation_coupling / total,
            dna_communication=self.dna_communication / total,
            interface_persistence=self.interface_persistence / total,
        )


@dataclass
class CooperativityScoreComponents:
    energetic_coupling: float
    dynamic_coupling: float
    orientation_coupling: float
    dna_communication: float
    interface_persistence: float


@dataclass
class CooperativityScoreResult:
    total_score: float
    components: CooperativityScoreComponents
    weights: CooperativityScoreWeights


def compute_cooperativity_score(
    components: CooperativityScoreComponents,
    weights: CooperativityScoreWeights | None = None,
) -> CooperativityScoreResult:
    """Composite, decomposable Cooperativity Score (section 22).

    Every component must already be normalised to [0, 1] by the caller
    (see :mod:`enhancoai.features.interaction` for how each component is
    derived from raw measurements). This function performs only the
    transparent weighted sum -- it is a heuristic composite indicator, not
    a validated biological constant (see section 22 of the spec).
    """
    weights = (weights or CooperativityScoreWeights()).normalised()
    for name, value in components.__dict__.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Cooperativity score component '{name}' must be in [0, 1], got {value}.")

    total = (
        weights.energetic_coupling * components.energetic_coupling
        + weights.dynamic_coupling * components.dynamic_coupling
        + weights.orientation_coupling * components.orientation_coupling
        + weights.dna_communication * components.dna_communication
        + weights.interface_persistence * components.interface_persistence
    )
    return CooperativityScoreResult(total_score=total, components=components, weights=weights)
