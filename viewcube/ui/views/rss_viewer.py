import numpy as np
import pyqtgraph as pg
from .base_viewer import BaseViewer
from PyQt5.QtCore import Qt


class RSSViewer(BaseViewer):
    def __init__(self, rss_data, parent=None):
        self.rss = rss_data
        super().__init__(parent)

    def _initUI(self):
        self.layout = pg.GraphicsLayoutWidget()
        self.view = self.layout.addViewBox()
        self.img = pg.ImageItem()
        self.view.addItem(self.img)

        # Configurar imagen RSS
        self.img.setImage(self.rss.data.T)
        self.view.setAspectLocked(False)

        # Configurar interacción
        self.view.setMouseMode(pg.ViewBox.PanMode)
        self.proxy = pg.SignalProxy(self.view.scene().sigMouseMoved, rateLimit=60,
                                    slot=self.mouseMoved)

    def _connectSignals(self):
        self.view.scene().sigMouseClicked.connect(self.mouseClicked)

    def updateDisplay(self, data):
        self.img.setImage(data.T)

    def setColorMap(self, cmap):
        self.img.setColorMap(pg.ColorMap(*zip(*cmap)))

    def setDynamicRange(self, vmin, vmax):
        self.img.setLevels((vmin, vmax))

    def mouseMoved(self, evt):
        pos = evt[0]
        if self.view.sceneBoundingRect().contains(pos):
            mousePoint = self.view.mapSceneToView(pos)
            x, y = int(mousePoint.x()), int(mousePoint.y())
            if 0 <= x < self.rss.data.shape[1] and 0 <= y < self.rss.data.shape[0]:
                self.spaxelSelected.emit(x, y)

    def mouseClicked(self, evt):
        if evt.button() == Qt.LeftButton:
            pos = evt.scenePos()
            mousePoint = self.view.mapSceneToView(pos)
            x, y = int(mousePoint.x()), int(mousePoint.y())
            if 0 <= x < self.rss.data.shape[1] and 0 <= y < self.rss.data.shape[0]:
                self.spaxelSelected.emit(x, y)

    def cleanUp(self):
        self.view.clear()
        self.img.deleteLater()