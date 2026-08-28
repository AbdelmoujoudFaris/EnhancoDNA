"""Visualisation tab (sections 44-47): 3D structure/contact/network rendering modes."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel

from enhancoai.app.state import ProjectState
from enhancoai.app.widgets.canvas import FigureCanvas

MODES = [
    "Protein",
    "DNA",
    "Protein-DNA contacts",
    "Allosteric network",
]


class VisualizationPanel(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODES)
        controls.addWidget(self.mode_combo)
        self.render_button = QPushButton("Render")
        self.render_button.clicked.connect(self._render)
        controls.addWidget(self.render_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.canvas = FigureCanvas()
        layout.addWidget(self.canvas, 1)

    def _render(self) -> None:
        mode = self.mode_combo.currentText()
        if self.state.structure is None:
            return

        from enhancoai.visualization.structure import render_structure_matplotlib
        from enhancoai.visualization.contacts import render_contact_map
        from enhancoai.visualization.allostery import render_network

        if mode == "Protein":
            figure = render_structure_matplotlib(self.state.structure, self.state.protein_chains, title="Protein")
        elif mode == "DNA":
            figure = render_structure_matplotlib(self.state.structure, self.state.dna_chains, title="DNA")
        elif mode == "Protein-DNA contacts":
            if self.state.protein_dna_result is None:
                return
            figure = render_contact_map(self.state.protein_dna_result.residue_contact_map)
        elif mode == "Allosteric network":
            if self.state.allostery_graph is None:
                return
            figure = render_network(self.state.allostery_graph)
        else:
            return

        self.canvas.set_figure(figure)
