"""HTML and PDF report generation (section 48)."""

from __future__ import annotations

from dataclasses import dataclass, field


REPORT_SECTION_TITLES = [
    "Structure",
    "TF/DNA Identification",
    "Protein-DNA Contacts",
    "Protein-Protein Contacts",
    "MD Analysis",
    "DNA Dynamics",
    "Orientation Analysis",
    "Free-Energy Analysis",
    "Cooperativity",
    "Allosteric Network",
    "AI Predictions",
    "Explainability",
    "Important Residues",
    "Important DNA Bases",
    "Limitations",
    "Reproducibility Information",
]


@dataclass
class ReportSection:
    title: str
    html_body: str = ""
    figure_paths: list[str] = field(default_factory=list)


@dataclass
class ReportData:
    project_name: str
    generated_at: str
    sections: list[ReportSection] = field(default_factory=list)
