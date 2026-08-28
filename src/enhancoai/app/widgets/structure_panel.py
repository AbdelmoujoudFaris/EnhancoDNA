"""Structure tab (section 8): load structure, classify chains, clean, select."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QGroupBox, QTextEdit, QSplitter,
)
from PySide6.QtCore import Qt

from enhancoai.app.state import ProjectState
from enhancoai.app.widgets.canvas import FigureCanvas


class StructurePanel(QWidget):
    def __init__(self, state: ProjectState, on_structure_loaded=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.on_structure_loaded = on_structure_loaded

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.load_button = QPushButton("Load Structure (PDB/mmCIF)...")
        self.load_button.clicked.connect(self._load_structure)
        controls.addWidget(self.load_button)
        self.path_label = QLabel("No structure loaded.")
        controls.addWidget(self.path_label, 1)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)

        chain_group = QGroupBox("Chains")
        chain_layout = QVBoxLayout(chain_group)
        self.chain_table = QTableWidget(0, 4)
        self.chain_table.setHorizontalHeaderLabels(["Chain", "Type", "Residues", "Atoms"])
        chain_layout.addWidget(self.chain_table)

        self.report_box = QTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setPlaceholderText("Missing-atom / cleaning report appears here.")
        chain_layout.addWidget(self.report_box)
        splitter.addWidget(chain_group)

        viewer_group = QGroupBox("3D Molecular Viewer")
        viewer_layout = QVBoxLayout(viewer_group)
        self.canvas = FigureCanvas()
        viewer_layout.addWidget(self.canvas)
        splitter.addWidget(viewer_group)

        splitter.setSizes([350, 550])
        layout.addWidget(splitter, 1)

    def _load_structure(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Structure", "", "Structure files (*.pdb *.ent *.cif *.mmcif);;All files (*)"
        )
        if not path:
            return
        self._load_from_path(path)

    def _load_from_path(self, path: str) -> None:
        from enhancoai.structure.parser import load_structure
        from enhancoai.structure.chain_detection import summarize_chains, classify_chains, ChainType
        from enhancoai.structure.cleaning import report_missing_backbone_atoms
        from enhancoai.visualization.structure import render_structure_matplotlib

        structure = load_structure(path)
        classification = classify_chains(structure)

        self.state.structure_path = path
        self.state.structure = structure
        self.state.chain_classification = classification
        self.state.protein_chains = [c for c, t in classification.items() if t == ChainType.PROTEIN]
        self.state.dna_chains = [c for c, t in classification.items() if t == ChainType.DNA]

        self.path_label.setText(path)
        self._populate_chain_table(summarize_chains(structure))

        missing = report_missing_backbone_atoms(structure)
        self.report_box.setPlainText("\n".join(missing) if missing else "No missing backbone atoms detected.")

        figure = render_structure_matplotlib(structure, title=f"{len(structure.chain_ids)} chain(s)")
        self.canvas.set_figure(figure)

        if self.on_structure_loaded:
            self.on_structure_loaded()

    def _populate_chain_table(self, summary_df) -> None:
        self.chain_table.setRowCount(len(summary_df))
        for row, record in enumerate(summary_df.itertuples()):
            self.chain_table.setItem(row, 0, QTableWidgetItem(str(record.chain_id)))
            self.chain_table.setItem(row, 1, QTableWidgetItem(str(record.type)))
            self.chain_table.setItem(row, 2, QTableWidgetItem(str(record.n_residues)))
            self.chain_table.setItem(row, 3, QTableWidgetItem(str(record.n_atoms)))
