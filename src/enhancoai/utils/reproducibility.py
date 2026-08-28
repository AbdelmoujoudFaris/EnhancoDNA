"""Reproducibility utilities: seeding, hashing, and reproducibility-package export.

EnhancoAI never presents a result without recording enough metadata to
reproduce it (input hashes, software versions, random seed, timestamp).
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs for reproducible runs."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def file_hash(path: str | Path, algorithm: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """Compute a content hash of a file for provenance tracking."""
    path = Path(path)
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def software_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("torch", "numpy", "scipy", "pandas", "sklearn", "MDAnalysis", "Bio", "biotite", "pyvista", "PySide6"):
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    try:
        import torch

        versions["cuda"] = torch.version.cuda or "not available"
    except ImportError:
        versions["cuda"] = "unknown"
    return versions


@dataclass
class ReproducibilityRecord:
    input_hashes: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    software_versions: dict[str, str] = field(default_factory=software_versions)
    seed: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_input(self, name: str, path: str | Path) -> None:
        self.input_hashes[name] = file_hash(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_hashes": self.input_hashes,
            "parameters": self.parameters,
            "software_versions": self.software_versions,
            "seed": self.seed,
            "timestamp": self.timestamp,
        }

    def export(self, path: str | Path) -> None:
        """Export the reproducibility package as JSON ("Export Reproducibility Package")."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
