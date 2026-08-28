"""Matplotlib-figure-in-Qt canvas, used for the 3D viewer and all 2D plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt


class FigureCanvas(FigureCanvasQTAgg):
    def __init__(self, figure=None):
        self._figure = figure or plt.figure()
        super().__init__(self._figure)

    def set_figure(self, figure) -> None:
        old_figure = self.figure
        self.figure = figure
        self.draw()
        plt.close(old_figure)
