"""MD trajectory loading via MDAnalysis.

Supports .xtc, .trr, .dcd, .nc, .pdb, .gro (any format MDAnalysis's
readers support). MDAnalysis is a required dependency for this module;
if it is unavailable a clear ImportError is raised rather than silently
producing empty results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrajectoryHandle:
    universe: "object"
    topology_path: str
    trajectory_path: str | None
    n_frames: int
    stride: int


def load_trajectory(
    topology_path: str | Path,
    trajectory_path: str | Path | None = None,
    stride: int = 1,
) -> TrajectoryHandle:
    """Load a topology (+ optional trajectory) with MDAnalysis.

    If ``trajectory_path`` is None, the topology file itself is treated as
    a single-frame "trajectory" (e.g. a static PDB).
    """
    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise ImportError(
            "MD trajectory analysis requires MDAnalysis. Install it via "
            "`pip install MDAnalysis` or `conda install -c conda-forge mdanalysis`."
        ) from exc

    topology_path = Path(topology_path)
    if not topology_path.exists():
        raise FileNotFoundError(f"Topology file not found: {topology_path}")

    if trajectory_path is not None:
        trajectory_path = Path(trajectory_path)
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
        universe = mda.Universe(str(topology_path), str(trajectory_path))
    else:
        universe = mda.Universe(str(topology_path))

    n_frames = len(universe.trajectory)
    return TrajectoryHandle(
        universe=universe,
        topology_path=str(topology_path),
        trajectory_path=str(trajectory_path) if trajectory_path else None,
        n_frames=n_frames,
        stride=stride,
    )


def iter_frames(handle: TrajectoryHandle):
    """Yield frame indices according to the configured stride."""
    for ts in handle.universe.trajectory[:: handle.stride]:
        yield ts.frame
