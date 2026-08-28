# Contributing to EnhancoAI

Thanks for considering a contribution. EnhancoAI is a general-purpose
platform for transcription-factor/DNA cooperativity analysis -- please keep
contributions general (not hard-coded to one biological system) and honest
about what is a validated calculation versus an exploratory/AI-based proxy
(see [`docs/reproducibility.md`](docs/reproducibility.md) and the
"No Fake Science" principle below).

## Development setup

```bash
git clone <repository>
cd EnhancoAI
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,full]"
pre-commit install
```

## Running tests

```bash
python scripts/generate_demo_dataset.py   # needed by most fixtures
pytest
```

Tests must run without a GPU and without any external MD program (GROMACS,
OpenMM, PLUMED, DSSP, FreeSASA) installed; features that need those must
degrade gracefully (clear message, not a crash or a fabricated result).

## Code style

- Type hints and docstrings on public functions/classes.
- `black` + `ruff` (run via `pre-commit`).
- No hard-coded paths, no hidden parameters -- everything scientific goes
  through `enhancoai.utils.config`.

## No Fake Science

This is the one rule that overrides all others:

- Never fabricate MD trajectories, free energies, cooperativity values, or
  model performance.
- If a model has no trained weights, say so (`Predictor` already does this
  -- follow that pattern).
- Label every AI-derived perturbation result as "AI-based in-silico
  perturbation" unless a real simulation was run.
- If an analysis is exploratory or under-sampled, say so in the result, not
  just in a comment.

## Pull requests

- Keep PRs focused; one module/feature per PR where possible.
- Add or update tests for anything you change under `src/enhancoai/`.
- Update `CHANGELOG.md` under "Unreleased".
