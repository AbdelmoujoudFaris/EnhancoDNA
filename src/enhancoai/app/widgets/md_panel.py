"""MD tab (section 13): trajectory loading, RMSD/RMSF."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QSplitter,
)
from PySide6.QtCore import Qt

from enhancoai.app.state import ProjectState
from enhancoai.app.widgets.canvas import FigureCanvas


class MDPanel(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.topology_edit = QLineEdit()
        self.topology_edit.setPlaceholderText("Topology (PDB/GRO/...) -- required")
        browse_topology = QPushButton("Browse Topology...")
        browse_topology.clicked.connect(self._browse_topology)
        self.trajectory_edit = QLineEdit()
        self.trajectory_edit.setPlaceholderText("Trajectory (XTC/DCD/...) -- optional")
        browse_trajectory = QPushButton("Browse Trajectory...")
        browse_trajectory.clicked.connect(self._browse_trajectory)
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self._load)

        controls.addWidget(self.topology_edit)
        controls.addWidget(browse_topology)
        controls.addWidget(self.trajectory_edit)
        controls.addWidget(browse_trajectory)
        controls.addWidget(self.load_button)
        layout.addLayout(controls)

        analysis_controls = QHBoxLayout()
        self.rmsd_button = QPushButton("Compute RMSD (protein)")
        self.rmsd_button.clicked.connect(lambda: self._plot_rmsd("protein and name CA"))
        self.rmsf_button = QPushButton("Compute RMSF (protein)")
        self.rmsf_button.clicked.connect(lambda: self._plot_rmsf("protein and name CA"))
        analysis_controls.addWidget(self.rmsd_button)
        analysis_controls.addWidget(self.rmsf_button)
        analysis_controls.addStretch(1)
        layout.addLayout(analysis_controls)

        self.status_label = QLabel("No trajectory loaded.")
        layout.addWidget(self.status_label)

        self.canvas = FigureCanvas()
        layout.addWidget(self.canvas, 1)

    def _browse_topology(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Topology")
        if path:
            self.topology_edit.setText(path)

    def _browse_trajectory(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Trajectory")
        if path:
            self.trajectory_edit.setText(path)

    def _load(self) -> None:
        from enhancoai.md.loader import load_trajectory

        topology = self.topology_edit.text().strip()
        trajectory = self.trajectory_edit.text().strip() or None
        if not topology:
            self.status_label.setText("Provide a topology file.")
            return
        try:
            handle = load_trajectory(topology, trajectory)
        except (FileNotFoundError, ImportError) as exc:
            self.status_label.setText(str(exc))
            return
        self.state.trajectory_path = trajectory or topology
        self.state.trajectory_handle = handle
        self.status_label.setText(f"Loaded {handle.n_frames} frame(s).")

    def _plot_rmsd(self, selection: str) -> None:
        from enhancoai.md.rmsd import compute_rmsd
        from enhancoai.visualization.plots import plot_timeseries

        if self.state.trajectory_handle is None:
            self.status_label.setText("Load a trajectory first.")
            return
        rmsd = compute_rmsd(self.state.trajectory_handle, selection=selection)
        figure = plot_timeseries(rmsd, "time_ps", "rmsd", title="RMSD", ylabel="RMSD (A)")
        self.canvas.set_figure(figure)

    def _plot_rmsf(self, selection: str) -> None:
        from enhancoai.md.rmsf import compute_rmsf
        from enhancoai.visualization.plots import plot_rmsf

        if self.state.trajectory_handle is None:
            self.status_label.setText("Load a trajectory first.")
            return
        rmsf = compute_rmsf(self.state.trajectory_handle, selection=selection)
        figure = plot_rmsf(rmsf)
        self.canvas.set_figure(figure)
