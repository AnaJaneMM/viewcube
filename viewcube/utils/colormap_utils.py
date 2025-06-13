import numpy as np
import pyqtgraph as pg
from pyqtgraph import ColorMap
from PyQt5.QtCore import Qt
from typing import Optional, Tuple


class IntColorMapPyQt:
    """Clase para ajuste interactivo de mapas de color en PyQtGraph"""

    def __init__(self, image_item: pg.ImageItem, plot_widget: pg.PlotWidget):
        self.image_item = image_item
        self.plot_widget = plot_widget
        self.cmin, self.cmax = self.image_item.getLevels()

        # Configurar eventos
        self.proxy = self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_clicked)

    def on_mouse_clicked(self, event):
        """Maneja clics del mouse para ajustar niveles"""
        if self.image_item.isVisible() and event.button() in [Qt.LeftButton, Qt.RightButton]:
            pos = self.plot_widget.getViewBox().mapSceneToView(event.scenePos())
            if not self.plot_widget.getViewBox().viewRect().contains(pos):
                return

            current_levels = self.image_item.getLevels()
            val = self.image_item.getHistogram()[1][np.argmin(np.abs(self.image_item.getHistogram()[0] - pos.x()))]

            if event.button() == Qt.LeftButton:  # Ajustar mínimo
                self.cmin = val
            elif event.button() == Qt.RightButton:  # Ajustar máximo
                self.cmax = val

            self.image_item.setLevels((self.cmin, self.cmax))


def pg_norm(norm_type: str = 'sqrt') -> Tuple[Optional[ColorMap], Tuple[float, float]]:
    """Crea una normalización y mapa de colores para PyQtGraph"""
    pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    color = np.array([
        (0, 0, 0, 255),
        (0, 0, 255, 255),
        (0, 255, 0, 255),
        (255, 255, 0, 255),
        (255, 0, 0, 255)
    ], dtype=np.ubyte)

    if norm_type == 'sqrt':
        cmap = ColorMap(pos, color)
        return cmap, (0, 1)
    elif norm_type == 'log':
        cmap = ColorMap(pos, color)
        return cmap, (1e-3, 1)
    else:  # Linear
        cmap = ColorMap(pos, color)
        return cmap, None