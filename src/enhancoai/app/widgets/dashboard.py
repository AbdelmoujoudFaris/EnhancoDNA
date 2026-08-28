"""Dashboard tab (section 7): summary of the current project state."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFormLayout, QGroupBox

from enhancoai.app.state import ProjectState


class DashboardWidget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        title = QLabel("EnhancoAI Dashboard")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.summary_group = QGroupBox("Project Summary")
        self.summary_layout = QFormLayout(self.summary_group)
        self._summary_labels: dict[str, QLabel] = {}
        for key in self.state.summary():
            value_label = QLabel("-")
            self._summary_labels[key] = value_label
            self.summary_layout.addRow(f"{key}:", value_label)
        layout.addWidget(self.summary_group)

        self.metrics_group = QGroupBox("Analysis Metrics")
        self.metrics_layout = QFormLayout(self.metrics_group)
        self._metric_labels: dict[str, QLabel] = {}
        for key in (
            "TFs", "Interface residues", "DNA-contact residues",
            "Cooperativity score", "Allosteric communication score", "AI prediction",
        ):
            value_label = QLabel("-")
            value_label.setObjectName("MetricValue")
            self._metric_labels[key] = value_label
            self.metrics_layout.addRow(f"{key}:", value_label)
        layout.addWidget(self.metrics_group)
        layout.addStretch(1)

    def refresh(self) -> None:
        for key, value in self.state.summary().items():
            if key in self._summary_labels:
                self._summary_labels[key].setText(str(value))

        n_tfs = len(self.state.protein_chains)
        self._metric_labels["TFs"].setText(str(n_tfs))

        if self.state.protein_dna_result is not None:
            summary = self.state.protein_dna_result.summary()
            self._metric_labels["Interface residues"].setText(str(summary["n_interface_protein_residues"]))
            self._metric_labels["DNA-contact residues"].setText(str(summary["n_contacted_dna_bases"]))

        if self.state.cooperativity_score is not None:
            self._metric_labels["Cooperativity score"].setText(f"{self.state.cooperativity_score.total_score:.2f}")
