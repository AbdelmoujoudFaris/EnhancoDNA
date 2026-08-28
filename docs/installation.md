# Installation

## Requirements

- Python >= 3.10
- A C/C++ toolchain is not required (all core dependencies ship wheels).

## Conda (recommended)

```bash
git clone <repository>
cd EnhancoAI
conda env create -f environment.yml
conda activate enhancoai
pip install -e .
```

## Plain pip / venv

```bash
python -m venv .venv
source .venv/bin/activate       # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Optional extras

```bash
pip install -e ".[full]"   # biotite, pyvista, pyvistaqt, torch_geometric
```

EnhancoAI works without these: the DNA module falls back to its built-in
simplified geometry calculations without `biotite`, 3D visualisation falls
back to matplotlib without `pyvista`, and the GNN falls back to a
pure-PyTorch message-passing layer without `torch_geometric`.

## External MD/structure tools (optional, never required)

| Tool | Used for | Fallback if absent |
|---|---|---|
| GROMACS / OpenMM / PLUMED | Running MD (not performed by EnhancoAI itself) | Load trajectories produced elsewhere via `enhancoai md` |
| DSSP (`mkdssp`) | Secondary-structure fractions | `secondary_structure_fractions()` returns `None`, not fabricated values |
| FreeSASA | Alternative SASA backend | Biopython's `ShrakeRupley` SASA is used instead |

## Verifying the install

```bash
python -m compileall src
python -m enhancoai --help
python scripts/generate_demo_dataset.py
pytest
```

## GPU

PyTorch is installed CPU-only by default via `environment.yml`/
`requirements.txt`. For CUDA, install a matching `torch` build from
https://pytorch.org/get-started/locally/ *after* the base install; do not
edit `pyproject.toml` to pin a CUDA build, since that breaks CPU-only CI
and Docker.

## Docker

```bash
docker build -t enhancoai .
docker run --rm enhancoai structure --input /data/complex.pdb
```

The default image is CPU-only and runs Qt in offscreen mode (headless);
mount your data directory with `-v $(pwd)/data:/data`.
