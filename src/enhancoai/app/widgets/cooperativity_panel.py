"""Cooperativity tab (sections 18, 22): composite Cooperativity Score."""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel,
    QDoubleSpinBox, QGroupBox,
)

from enhancoai.app.state import ProjectState
from enhancoai.app.widgets.canvas import FigureCanvas
from enhancoai.free_energy.cooperativity import compute_cooperativity_score, CooperativityScoreWeights
from enhancoai.features.interaction import CooperativityEvidence, evidence_to_components


class CooperativityPanel(QWidget):
    """Manual-evidence-entry cooperativity score calculator.

    Raw measurements (energetic shift, dynamic shift, ...) are typically
    produced by the MD/free-energy/allostery pipelines run from the other
    tabs or the CLI; this panel lets a user enter or paste those numbers
    directly and see the transparent, decomposable score they produce.
    """

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QHBoxLayout(self)

        form_group = QGroupBox("Cooperativity Evidence (raw measurements)")
        form = QFormLayout(form_group)
        self.energetic_spin = self._make_spin(0.0, 20.0, 0.5, " kcal/mol")
        self.dynamic_spin = self._make_spin(0.0, 10.0, 0.1, " A")
        self.orientation_spin = self._make_spin(0.0, 180.0, 1.0, " deg")
        self.dna_pathway_spin = self._make_spin(0.0, 1.0, 0.01, "")
        self.interface_spin = self._make_spin(0.0, 1.0, 0.01, "")

        form.addRow("|delta_delta_G_coop|:", self.energetic_spin)
        form.addRow("Delta RMSF(TF-A):", self.dynamic_spin)
        form.addRow("Delta orientation angle:", self.orientation_spin)
        form.addRow("DNA pathway strength [0-1]:", self.dna_pathway_spin)
        form.addRow("Interface persistence [0-1]:", self.interface_spin)

        self.compute_button = QPushButton("Compute Cooperativity Score")
        self.compute_button.clicked.connect(self._compute)
        form.addRow(self.compute_button)

        layout.addWidget(form_group)

        result_group = QGroupBox("Score Breakdown")
        result_layout = QVBoxLayout(result_group)
        self.total_label = QLabel("Total score: -")
        self.total_label.setObjectName("MetricValue")
        result_layout.addWidget(self.total_label)
        self.canvas = FigureCanvas()
        result_layout.addWidget(self.canvas)
        layout.addWidget(result_group, 1)

    @staticmethod
    def _make_spin(minimum, maximum, step, suffix) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        return spin

    def _compute(self) -> None:
        evidence = CooperativityEvidence(
            energetic_shift=self.energetic_spin.value(),
            dynamic_shift=self.dynamic_spin.value(),
            orientation_shift=self.orientation_spin.value(),
            dna_pathway_strength=self.dna_pathway_spin.value(),
            interface_persistence=self.interface_spin.value(),
        )
        components = evidence_to_components(evidence)
        result = compute_cooperativity_score(components, CooperativityScoreWeights())
        self.state.cooperativity_score = result

        self.total_label.setText(f"Total score: {result.total_score:.2f}")

        names = ["Energetic", "Dynamic", "Orientation", "DNA comm.", "Interface"]
        values = [
            result.components.energetic_coupling,
            result.components.dynamic_coupling,
            result.components.orientation_coupling,
            result.components.dna_communication,
            result.components.interface_persistence,
        ]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(names, values, color="#2ca02c")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Component value [0-1]")
        ax.set_title("Cooperativity Score components (heuristic composite indicator)")
        fig.tight_layout()
        self.canvas.set_figure(fig)
