import numpy as np
import pyqtgraph as pg
from .base_viewer import BaseViewer
from PyQt5.QtCore import Qt


class SpaxelViewer(BaseViewer):
    def __init__(self, cube_data, parent=None):
        self.cube = cube_data
        self.current_slice = 0
        super().__init__(parent)

    def _initUI(self):
        self.layout = pg.GraphicsLayoutWidget()
        self.view = self.layout.addViewBox()
        self.img = pg.ImageItem()
        self.view.addItem(self.img)

        # Configurar imagen inicial
        self.img.setImage(self.cube.data[self.current_slice])
        self.view.setAspectLocked(True)

        # Configurar interacción
        self.view.setMouseMode(pg.ViewBox.PanMode)
        self.proxy = pg.SignalProxy(self.view.scene().sigMouseMoved, rateLimit=60,
                                    slot=self.mouseMoved)

    def _connectSignals(self):
        self.view.scene().sigMouseClicked.connect(self.mouseClicked)

    def updateDisplay(self, data):
        self.img.setImage(data)

    def setColorMap(self, cmap):
        self.img.setColorMap(pg.ColorMap(*zip(*cmap)))

    def setDynamicRange(self, vmin, vmax):
        self.img.setLevels((vmin, vmax))

    def mouseMoved(self, evt):
        pos = evt[0]
        if self.view.sceneBoundingRect().contains(pos):
            mousePoint = self.view.mapSceneToView(pos)
            x, y = int(mousePoint.x()), int(mousePoint.y())
            if 0 <= x < self.cube.shape[2] and 0 <= y < self.cube.shape[1]:
                self.spaxelSelected.emit(x, y)

    def mouseClicked(self, evt):
        if evt.button() == Qt.LeftButton:
            pos = evt.scenePos()
            mousePoint = self.view.mapSceneToView(pos)
            x, y = int(mousePoint.x()), int(mousePoint.y())
            if 0 <= x < self.cube.shape[2] and 0 <= y < self.cube.shape[1]:
                self.spaxelSelected.emit(x, y)

    def changeSlice(self, slice_idx):
        self.current_slice = slice_idx
        self.updateDisplay(self.cube.data[self.current_slice])

    def cleanUp(self):
        self.view.clear()
        self.img.deleteLater()