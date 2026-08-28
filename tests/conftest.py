import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXAMPLES_DIR = REPO_ROOT / "data" / "examples"


@pytest.fixture(scope="session")
def demo_two_factor_pdb() -> Path:
    path = EXAMPLES_DIR / "demo_tf_dna_two_factor.pdb"
    if not path.exists():
        pytest.skip(
            "Demo dataset not generated. Run `python scripts/generate_demo_dataset.py` first."
        )
    return path


@pytest.fixture(scope="session")
def demo_single_factor_pdb() -> Path:
    path = EXAMPLES_DIR / "demo_tf_dna_single_factor.pdb"
    if not path.exists():
        pytest.skip(
            "Demo dataset not generated. Run `python scripts/generate_demo_dataset.py` first."
        )
    return path


@pytest.fixture()
def demo_structure(demo_two_factor_pdb):
    from enhancoai.structure.parser import load_structure

    return load_structure(demo_two_factor_pdb)
