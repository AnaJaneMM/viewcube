import pyqtgraph as pg
from .base_viewer import BaseViewer
from PyQt5.QtCore import Qt


class SpectrumViewer(BaseViewer):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.spectra = {}

    def _initUI(self):
        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'Flux', units='erg/s/cm²/Å')
        self.plot.setLabel('bottom', 'Wavelength', units='Å')
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True)

        # Selector de rango espectral
        self.range = pg.LinearRegionItem()
        self.range.setZValue(10)
        self.plot.addItem(self.range)

    def _connectSignals(self):
        self.range.sigRegionChanged.connect(self.rangeChanged)

    def addSpectrum(self, wavelength, flux, name='Spectrum', color='b'):
        curve = self.plot.plot(wavelength, flux, name=name, pen=color)
        self.spectra[name] = curve

    def updateSpectrum(self, name, wavelength, flux):
        if name in self.spectra:
            self.spectra[name].setData(wavelength, flux)

    def setRange(self, wmin, wmax):
        self.range.setRegion((wmin, wmax))

    def rangeChanged(self):
        self.wavelengthRangeChanged.emit(*self.range.getRegion())

    def cleanUp(self):
        self.plot.clear()