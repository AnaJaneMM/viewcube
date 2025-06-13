from abc import ABC, abstractmethod
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal


class BaseViewer(QWidget, ABC):
    dataUpdated = pyqtSignal(object)
    spaxelSelected = pyqtSignal(int, int)
    wavelengthRangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._initUI()
        self._connectSignals()

    @abstractmethod
    def _initUI(self):
        """Inicializar componentes de la UI"""
        pass

    @abstractmethod
    def _connectSignals(self):
        """Conectar señales internas"""
        pass

    @abstractmethod
    def updateDisplay(self, data):
        """Actualizar visualización con nuevos datos"""
        pass

    @abstractmethod
    def setColorMap(self, cmap):
        """Establecer mapa de colores"""
        pass

    @abstractmethod
    def setDynamicRange(self, vmin, vmax):
        """Ajustar rango dinámico"""
        pass

    @abstractmethod
    def clearSelection(self):
        """Limpiar selecciones"""
        pass

    @abstractmethod
    def cleanUp(self):
        """Liberar recursos"""
        pass