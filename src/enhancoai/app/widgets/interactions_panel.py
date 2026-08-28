"""Interactions tab (sections 9, 11): protein-DNA and protein-protein contact analysis."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QGroupBox, QSplitter, QTextEdit,
)
from PySide6.QtCore import Qt

from enhancoai.app.state import ProjectState
from enhancoai.app.widgets.canvas import FigureCanvas


class InteractionsPanel(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Protein chain:"))
        self.protein_combo = QComboBox()
        controls.addWidget(self.protein_combo)
        controls.addWidget(QLabel("DNA chain:"))
        self.dna_combo = QComboBox()
        controls.addWidget(self.dna_combo)
        self.run_button = QPushButton("Analyse Protein-DNA Contacts")
        self.run_button.clicked.connect(self._run_protein_dna)
        controls.addWidget(self.run_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        splitter.addWidget(self.summary_box)

        self.canvas = FigureCanvas()
        splitter.addWidget(self.canvas)
        splitter.setSizes([350, 550])
        layout.addWidget(splitter, 1)

    def refresh_chains(self) -> None:
        self.protein_combo.clear()
        self.protein_combo.addItems(self.state.protein_chains)
        self.dna_combo.clear()
        self.dna_combo.addItems(self.state.dna_chains)

    def _run_protein_dna(self) -> None:
        if self.state.structure is None:
            self.summary_box.setPlainText("Load a structure first (Structure tab).")
            return
        protein_chain = self.protein_combo.currentText()
        dna_chain = self.dna_combo.currentText()
        if not protein_chain or not dna_chain:
            self.summary_box.setPlainText("Select a protein and a DNA chain.")
            return

        from enhancoai.interactions.protein_dna import analyse_protein_dna_interactions, format_contact_descriptions
        from enhancoai.visualization.contacts import render_contact_map

        result = analyse_protein_dna_interactions(self.state.structure, protein_chain, dna_chain)
        self.state.protein_dna_result = result

        lines = [f"{k}: {v}" for k, v in result.summary().items()]
        lines.append("")
        lines.extend(format_contact_descriptions(result)[:50])
        self.summary_box.setPlainText("\n".join(lines))

        figure = render_contact_map(result.residue_contact_map, title=f"{protein_chain} <-> {dna_chain}")
        self.canvas.set_figure(figure)
