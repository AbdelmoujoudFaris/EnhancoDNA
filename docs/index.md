# EnhancoAI Documentation

EnhancoAI investigates how transcription factors (TFs) cooperatively
recognise DNA, how protein-protein interactions modify DNA recognition,
and how DNA itself can mediate allosteric communication between distant
binding sites. It is a general platform, not tied to any one TF pair.

## Where to start

- New here: [`installation.md`](installation.md) then [`quickstart.md`](quickstart.md).
- Why this exists: [`scientific_background.md`](scientific_background.md).
- Using the desktop app: [`gui.md`](gui.md).
- Loading and analysing a structure: [`structure_analysis.md`](structure_analysis.md).
- Trajectory analysis: [`md_analysis.md`](md_analysis.md).
- DNA geometry and allosteric networks: [`dna_allostery.md`](dna_allostery.md).
- Cooperativity metrics: [`cooperativity.md`](cooperativity.md).
- The four PyTorch models: [`ai_models.md`](ai_models.md).
- Explainable AI and perturbation: [`explainability.md`](explainability.md).
- Provenance and result caveats: [`reproducibility.md`](reproducibility.md).

## Design principle: no fake science

Every module in EnhancoAI follows the same rule: if a result cannot be
computed from what was supplied (no trajectory, no trained weights, too
few samples), the software says so explicitly rather than returning a
plausible-looking but fabricated number. Keep this in mind when reading
any output, and when contributing new analyses.
