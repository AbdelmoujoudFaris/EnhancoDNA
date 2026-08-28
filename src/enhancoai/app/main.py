"""Application entry point: `enhancoai gui` / `python -m enhancoai.app.main`."""

from __future__ import annotations

import sys


def run_app() -> int:
    from PySide6.QtWidgets import QApplication

    from enhancoai.app.main_window import MainWindow
    from enhancoai.app.styles import STYLESHEET

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
