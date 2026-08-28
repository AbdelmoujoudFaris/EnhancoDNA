"""EnhancoAI main window (section 6): tabbed workflow with a shared project state."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget, QStatusBar, QLabel
from PySide6.QtCore import Qt

from enhancoai import __version__
from enhancoai.app.state import ProjectState
from enhancoai.app.widgets.dashboard import DashboardWidget
from enhancoai.app.widgets.structure_panel import StructurePanel
from enhancoai.app.widgets.interactions_panel import InteractionsPanel
from enhancoai.app.widgets.md_panel import MDPanel
from enhancoai.app.widgets.cooperativity_panel import CooperativityPanel
from enhancoai.app.widgets.ai_panel import AIPanel
from enhancoai.app.widgets.visualization_panel import VisualizationPanel
from enhancoai.app.widgets.hypothesis_panel import HypothesisPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"EnhancoAI v{__version__} -- Transcription Factor Cooperativity & DNA Allostery")
        self.resize(1400, 900)

        self.state = ProjectState()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.dashboard = DashboardWidget(self.state)
        self.structure_panel = StructurePanel(self.state, on_structure_loaded=self._on_structure_loaded)
        self.interactions_panel = InteractionsPanel(self.state)
        self.md_panel = MDPanel(self.state)
        self.cooperativity_panel = CooperativityPanel(self.state)
        self.ai_panel = AIPanel(self.state)
        self.visualization_panel = VisualizationPanel(self.state)
        self.hypothesis_panel = HypothesisPanel(self.state)

        self.tabs.addTab(self.dashboard, "Project")
        self.tabs.addTab(self.structure_panel, "Structure")
        self.tabs.addTab(self.interactions_panel, "Interactions")
        self.tabs.addTab(self.md_panel, "MD")
        self.tabs.addTab(self.cooperativity_panel, "Cooperativity")
        self.tabs.addTab(self.ai_panel, "AI")
        self.tabs.addTab(self.visualization_panel, "Visualisation")
        self.tabs.addTab(self.hypothesis_panel, "Hypothesis Testing")

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.status_label = QLabel("Ready.")
        status_bar.addWidget(self.status_label)

        from enhancoai.utils.device import detect_device, describe

        status_bar.addPermanentWidget(QLabel(describe(detect_device())))

    def _on_structure_loaded(self) -> None:
        self.dashboard.refresh()
        self.interactions_panel.refresh_chains()
        self.status_label.setText(f"Structure loaded: {self.state.structure_path}")
