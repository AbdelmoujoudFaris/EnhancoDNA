"""Structure parsing for PDB and mmCIF files.

Produces a backend-agnostic :class:`StructureData` (a flat atom table) so
downstream modules never depend on Biopython internals directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ATOM_COLUMNS = [
    "atom_index",
    "atom_name",
    "element",
    "altloc",
    "res_name",
    "chain_id",
    "res_seq",
    "icode",
    "x",
    "y",
    "z",
    "occupancy",
    "b_factor",
    "is_hetero",
    "model",
]


@dataclass
class StructureData:
    """Flat, backend-agnostic representation of a molecular structure.

    Attributes
    ----------
    atoms:
        A pandas DataFrame with one row per atom and columns
        ``ATOM_COLUMNS``.
    source_path:
        Path the structure was loaded from.
    """

    atoms: pd.DataFrame
    source_path: str
    missing_atom_report: list[str] = field(default_factory=list)

    @property
    def coords(self) -> np.ndarray:
        return self.atoms[["x", "y", "z"]].to_numpy(dtype=float)

    @property
    def chain_ids(self) -> list[str]:
        return list(dict.fromkeys(self.atoms["chain_id"].tolist()))

    def chain(self, chain_id: str) -> pd.DataFrame:
        return self.atoms[self.atoms["chain_id"] == chain_id]

    def n_atoms(self) -> int:
        return len(self.atoms)

    def n_models(self) -> int:
        return int(self.atoms["model"].nunique())

    def residues(self, chain_id: str | None = None) -> pd.DataFrame:
        """Return one row per residue (first atom's metadata), optionally filtered by chain."""
        atoms = self.atoms if chain_id is None else self.chain(chain_id)
        return (
            atoms.drop_duplicates(subset=["model", "chain_id", "res_seq", "icode"])
            .loc[:, ["model", "chain_id", "res_seq", "icode", "res_name", "is_hetero"]]
            .reset_index(drop=True)
        )


def _biopython_parser(path: Path):
    from Bio.PDB import MMCIFParser, PDBParser

    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def load_structure(path: str | Path, structure_id: str | None = None) -> StructureData:
    """Load a PDB or mmCIF file into a :class:`StructureData`.

    Supports ``.pdb``, ``.ent``, ``.cif`` and ``.mmcif`` extensions via
    Biopython. Alternate locations, missing atoms, occupancy and B-factors
    are preserved for downstream cleaning.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    structure_id = structure_id or path.stem
    parser = _biopython_parser(path)
    structure = parser.get_structure(structure_id, str(path))

    rows = []
    atom_index = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag, res_seq, icode = residue.get_id()
                is_hetero = hetflag.strip() != ""
                for atom in residue:
                    x, y, z = atom.get_coord().tolist()
                    rows.append(
                        {
                            "atom_index": atom_index,
                            "atom_name": atom.get_name(),
                            "element": (atom.element or "").strip() or _guess_element(atom.get_name()),
                            "altloc": atom.get_altloc() if atom.get_altloc() != " " else "",
                            "res_name": residue.get_resname().strip(),
                            "chain_id": chain.id,
                            "res_seq": int(res_seq),
                            "icode": icode.strip(),
                            "x": x,
                            "y": y,
                            "z": z,
                            "occupancy": float(atom.get_occupancy() or 1.0),
                            "b_factor": float(atom.get_bfactor() or 0.0),
                            "is_hetero": is_hetero,
                            "model": int(model.id),
                        }
                    )
                    atom_index += 1

    if not rows:
        raise ValueError(f"No atoms parsed from structure file: {path}")

    atoms = pd.DataFrame(rows, columns=ATOM_COLUMNS)
    return StructureData(atoms=atoms, source_path=str(path))


_ELEMENT_GUESS = {"C": "C", "N": "N", "O": "O", "S": "S", "P": "P", "H": "H"}


def _guess_element(atom_name: str) -> str:
    """Fallback element guess when the file omits the element column."""
    stripped = atom_name.strip().lstrip("0123456789")
    return _ELEMENT_GUESS.get(stripped[:1], stripped[:1] or "X")
