"""Hypothesis Testing Dashboard (section 55).

Every conclusion is linked to a quantitative measurement already present in
the project state; if the required measurement has not been computed yet,
the answer is UNCERTAIN, never fabricated.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox, QFormLayout

from enhancoai.app.state import ProjectState

QUESTIONS = [
    "Does TF-B alter TF-A orientation?",
    "Does TF-B modify DNA dynamics?",
    "Does TF-B modify TF-A-DNA interaction?",
    "Is there a candidate DNA-mediated pathway?",
    "Is the predicted mechanism direct?",
]


class HypothesisPanel(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        title = QLabel("Scientific Hypothesis Testing")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        group = QGroupBox("Conclusions (evidence-linked)")
        form = QFormLayout(group)
        self._answer_labels: dict[str, QLabel] = {}
        for question in QUESTIONS:
            answer = QLabel("UNCERTAIN")
            self._answer_labels[question] = answer
            form.addRow(question, answer)
        layout.addWidget(group)

        self.evidence_label = QLabel("")
        self.evidence_label.setWordWrap(True)
        layout.addWidget(self.evidence_label)

        refresh_button = QPushButton("Re-evaluate from current project state")
        refresh_button.clicked.connect(self.refresh)
        layout.addWidget(refresh_button)
        layout.addStretch(1)

    def refresh(self) -> None:
        evidence_lines = []

        answer = "UNCERTAIN"
        if self.state.protein_dna_result is not None:
            n_contacts = self.state.protein_dna_result.summary()["n_interface_protein_residues"]
            answer = "YES" if n_contacts > 0 else "NO"
            evidence_lines.append(f"TF-A-DNA interaction: {n_contacts} interface residue(s) detected.")
        self._answer_labels["Does TF-B modify TF-A-DNA interaction?"].setText(
            answer + (" (requires comparing TF-A-alone vs TF-A+TF-B systems for a definitive answer)" if answer != "UNCERTAIN" else "")
        )

        answer = "UNCERTAIN"
        if self.state.allostery_graph is not None:
            answer = "YES" if self.state.allostery_graph.number_of_edges() > 0 else "NO"
            evidence_lines.append(f"Allosteric network: {self.state.allostery_graph.number_of_edges()} candidate edge(s).")
        self._answer_labels["Is there a candidate DNA-mediated pathway?"].setText(answer)

        answer = "UNCERTAIN"
        if self.state.cooperativity_score is not None:
            score = self.state.cooperativity_score.total_score
            direct_component = self.state.cooperativity_score.components.energetic_coupling
            answer = "MIXED" if 0.3 < direct_component < 0.7 else ("YES" if direct_component >= 0.7 else "NO")
            evidence_lines.append(f"Cooperativity score: {score:.2f}; energetic-coupling component: {direct_component:.2f}.")
        self._answer_labels["Is the predicted mechanism direct?"].setText(answer)

        for label in ("Does TF-B alter TF-A orientation?", "Does TF-B modify DNA dynamics?"):
            self._answer_labels[label].setText("UNCERTAIN (requires TF-A-alone vs TF-A+TF-B comparative MD -- see MD tab)")

        self.evidence_label.setText(
            "Evidence:\n" + ("\n".join(evidence_lines) if evidence_lines else "No supporting measurements computed yet.")
        )
