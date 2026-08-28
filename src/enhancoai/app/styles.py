"""Modern scientific dark theme stylesheet for the EnhancoAI desktop app."""

STYLESHEET = """
QMainWindow, QWidget { background-color: #1e2228; color: #e6e6e6; font-family: 'Segoe UI', Arial; font-size: 10.5pt; }
QTabWidget::pane { border: 1px solid #333944; }
QTabBar::tab { background: #262b33; padding: 8px 16px; border: 1px solid #333944; }
QTabBar::tab:selected { background: #34495e; color: white; }
QGroupBox { border: 1px solid #333944; border-radius: 4px; margin-top: 1.2em; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #7fb3ff; }
QPushButton { background-color: #2c3e50; border: 1px solid #3d566e; border-radius: 4px; padding: 6px 14px; }
QPushButton:hover { background-color: #34495e; }
QPushButton:disabled { background-color: #22262c; color: #666; }
QTableWidget, QTableView { background-color: #23272e; gridline-color: #333944; }
QHeaderView::section { background-color: #2c3138; padding: 4px; border: none; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #23272e; border: 1px solid #3d566e; border-radius: 3px; padding: 3px; }
QStatusBar { background-color: #161a1f; }
QLabel#SectionTitle { font-size: 13pt; font-weight: bold; color: #7fb3ff; }
QLabel#MetricValue { font-size: 16pt; font-weight: bold; }
"""
