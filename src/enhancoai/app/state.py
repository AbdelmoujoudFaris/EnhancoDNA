"""Shared, mutable project state passed between GUI tabs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectState:
    project_name: str = "Untitled Project"
    structure_path: str | None = None
    structure: object | None = None  # enhancoai.structure.parser.StructureData
    chain_classification: dict | None = None
    protein_chains: list[str] = field(default_factory=list)
    dna_chains: list[str] = field(default_factory=list)

    trajectory_path: str | None = None
    trajectory_handle: object | None = None

    protein_dna_result: object | None = None
    protein_protein_result: object | None = None

    cooperativity_score: object | None = None
    allostery_graph: object | None = None

    def summary(self) -> dict:
        return {
            "Project": self.project_name,
            "Structure": self.structure_path or "(none loaded)",
            "Protein chains": ", ".join(self.protein_chains) or "-",
            "DNA chains": ", ".join(self.dna_chains) or "-",
            "Trajectory": self.trajectory_path or "(none loaded)",
            "Frames": self.trajectory_handle.n_frames if self.trajectory_handle else "-",
        }
