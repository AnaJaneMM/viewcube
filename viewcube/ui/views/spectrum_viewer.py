"""Visualizador especializado para espectros."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, List, Callable


class SpectrumViewer(ViewerInterface):
    """Visualizador especializado para espectros."""

    def __init__(self, figure_size: Tuple[float, float] = (8, 5)):
        self.figure_size = figure_size
        self.figure = None
        self.axes = None

        # Límites de visualización
        self.wavelength_limits: Optional[Tuple[float, float]] = None
        self.flux_limits: Optional[Tuple[float, float]] = None

        # Datos actuales
        self.current_spectrum: Optional[np.ndarray] = None
        self.current_wavelength: Optional[np.ndarray] = None
        self.comparison_spectrum: Optional[np.ndarray] = None
        self.comparison_wavelength: Optional[np.ndarray] = None

        # Líneas de plot
        self.spectrum_line = None
        self.comparison_line = None
        self.filter_line = None
        self.error_bars = None

        # Callbacks
        self.on_click_callback: Optional[Callable] = None

        self.setup_figure()

    def setup_figure(self) -> None:
        """Configura la figura y ejes para espectros."""
        self.figure = plt.figure(2, self.figure_size)
        self.figure.set_label("Spectral Viewer")
        self.figure.canvas.manager.set_window_title("Spectral Viewer")
        self.axes = self.figure.add_subplot(111)

        # Conectar eventos
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)

    def plot_spectrum(self, wavelength: np.ndarray, spectrum: np.ndarray,
                      color: str = "#1f77b4", linewidth: float = 1.0,
                      label: str = "Spectrum") -> None:
        """Plotea espectro principal."""
        self.current_wavelength = wavelength
        self.current_spectrum = spectrum

        if self.spectrum_line:
            self.spectrum_line.remove()

        self.spectrum_line = self.axes.plot(wavelength, spectrum,
                                            color=color, linewidth=linewidth,
                                            label=label)[0]
        self._update_limits()
        self.figure.canvas.draw()

    def plot_comparison(self, wavelength: np.ndarray, spectrum: np.ndarray,
                        color: str = "#ff7f0e", linewidth: float = 1.0,
                        label: str = "Comparison") -> None:
        """Plotea espectro de comparación."""
        self.comparison_wavelength = wavelength
        self.comparison_spectrum = spectrum

        if self.comparison_line:
            self.comparison_line.remove()

        self.comparison_line = self.axes.plot(wavelength, spectrum,
                                              color=color, linewidth=linewidth,
                                              label=label)[0]
        self._update_limits()
        self.figure.canvas.draw()

    def add_error_bars(self, wavelength: np.ndarray, spectrum: np.ndarray,
                       errors: np.ndarray, color: str = "grey") -> None:
        """Añade barras de error."""
        if self.error_bars:
            self.error_bars.remove()

        self.error_bars = self.axes.errorbar(wavelength, spectrum, yerr=errors,
                                             fmt="none", ecolor=color)
        self.figure.canvas.draw()

    def plot_filter(self, wavelength: np.ndarray, filter_response: np.ndarray,
                    scale_factor: float = 1.0, color: str = "green") -> None:
        """Plotea respuesta del filtro."""
        if self.filter_line:
            self.filter_line.remove()

        scaled_response = filter_response * scale_factor
        self.filter_line = self.axes.plot(wavelength, scaled_response,
                                          color=color, alpha=0.7)[0]

        # Añadir área sombreada
        self.axes.fill_between(wavelength, scaled_response,
                               color=color, alpha=0.25)
        self.figure.canvas.draw()

    def set_wavelength_limits(self, wl_min: float, wl_max: float) -> None:
        """Establece límites de longitud de onda."""
        self.wavelength_limits = (wl_min, wl_max)
        self.axes.set_xlim(self.wavelength_limits)
        self.figure.canvas.draw()

    def set_flux_limits(self, flux_min: float, flux_max: float) -> None:
        """Establece límites de flujo."""
        self.flux_limits = (flux_min, flux_max)
        self.axes.set_ylim(self.flux_limits)
        self.figure.canvas.draw()

    def _update_limits(self) -> None:
        """Actualiza límites automáticamente si no están establecidos."""
        if self.wavelength_limits:
            self.axes.set_xlim(self.wavelength_limits)

        if self.flux_limits:
            self.axes.set_ylim(self.flux_limits)

    def clear_display(self) -> None:
        """Limpia la visualización de espectros."""
        self.axes.clear()
        self.spectrum_line = None
        self.comparison_line = None
        self.filter_line = None
        self.error_bars = None
        self.figure.canvas.draw()

    def update_display(self) -> None:
        """Actualiza la visualización."""
        if self.current_spectrum is not None and self.current_wavelength is not None:
            self.plot_spectrum(self.current_wavelength, self.current_spectrum)

    def set_title(self, title: str) -> None:
        """Establece título del espectro."""
        self.axes.set_title(title)
        self.figure.canvas.draw()

    def add_legend(self) -> None:
        """Añade leyenda al plot."""
        self.axes.legend()
        self.figure.canvas.draw()

    def _on_click(self, event) -> None:
        """Maneja eventos de click."""
        if self.on_click_callback and event.inaxes == self.axes:
            self.on_click_callback(event)
