# Scientific Background

## Motivating system

The design case is the OCT4-SOX2 enhanceosome, but EnhancoAI is a general
platform: nothing in `src/enhancoai/` references OCT4 or SOX2 by name.
Any protein-DNA complex (single TF, TF pair, or larger assembly) can be
analysed the same way.

## Three mechanisms

- **Direct cooperativity**: Protein A physically contacts Protein B
  (`enhancoai.interactions.protein_protein`).
- **DNA-mediated allostery**: binding of Protein A changes DNA structure,
  dynamics, flexibility or electrostatics in a way that alters Protein B's
  binding, without A and B directly touching
  (`enhancoai.dna`, `enhancoai.allostery`).
- **Latent DNA specificity**: a TF may stabilise a DNA configuration that
  is only weakly recognised when the TF binds in isolation -- visible by
  comparing DNA geometry features between the single-TF and multi-TF
  systems (`enhancoai.features.dna`).

## Research questions (Q1-Q10)

| # | Question | Where to look |
|---|---|---|
| Q1 | Does TF-B change TF-A's DNA-bound orientation? | `enhancoai.md.orientation` |
| Q2 | Does TF-B alter TF-A conformational dynamics? | `enhancoai.md.rmsf`, `enhancoai.md.correlations` |
| Q3 | Does TF-B modify TF-A's binding free-energy landscape? | `enhancoai.free_energy.pmf`, `.cooperativity` |
| Q4 | Can DNA transmit an allosteric signal between TFs? | `enhancoai.allostery.network`, `.pathways` |
| Q5 | Which DNA regions mediate the communication? | `enhancoai.allostery.pathways`, `explainability.mapping` |
| Q6 | Which protein residues participate? | same, plus `models.hybrid` residue-importance head |
| Q7 | Does direct PPI explain the observed cooperativity? | compare `energetic_coupling` vs `dna_communication` score components |
| Q8 | Can cooperativity exist with weak/no direct contact? | `enhancoai.interactions.protein_protein` (min distance) vs cooperativity score |
| Q9 | Can ML predict cooperative vs non-cooperative pairs? | `enhancoai.models` (four architectures) |
| Q10 | Are there structural signatures of enhanceosomes? | `enhancoai.features.interaction.build_allosteric_fingerprint` |

The GUI's **Hypothesis Testing** tab turns a subset of these into
evidence-linked YES/NO/UNCERTAIN answers, never a bare conclusion.
