import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from astropy.io import fits
from ...core.interfaces.presenter_interfaces import SpectrumPresenterInterface
from ..repositories.fits_repository import FitsRepository


class SpectrumPresenter(SpectrumPresenterInterface):
    """Presentador para visualización de espectros astronómicos"""

    def __init__(self, colormap='viridis', linewidth=1.5, figsize=(10, 6)):
        self.colormap = colormap
        self.linewidth = linewidth
        self.figsize = figsize
        self._current_figure = None
        self._current_axes = None
        self._filter_patches = []

    def present_spectrum(self, spectrum_data, title=None,
                         wavelength_range=None, flux_range=None):
        """Visualiza un espectro individual"""
        fig, ax = self._create_figure()

        wavelength = spectrum_data.wavelength
        flux = spectrum_data.flux
        error = spectrum_data.error
        flag = spectrum_data.flag

        # Aplicar máscara de flags
        if flag is not None:
            flux = np.ma.masked_where(flag, flux)
            if error is not None:
                error = np.ma.masked_where(flag, error)

        # Línea principal del espectro
        main_line = ax.plot(wavelength, flux,
                            lw=self.linewidth,
                            color=self.colormap,
                            label='Espectro observado')[0]

        # Manejo de errores
        if error is not None:
            self._plot_errors(ax, wavelength, flux, error)

        # Configuración de ejes
        self._configure_axes(ax, wavelength, flux,
                             wavelength_range, flux_range)

        # Metadatos
        if title:
            ax.set_title(title)

        self._add_metadata_annotations(ax, spectrum_data.meta)

        return fig

    def present_comparison(self, spectrum1, spectrum2,
                           labels=('Observado', 'Modelo')):
        """Compara dos espectros"""
        fig, ax = self._create_figure()

        # Primer espectro
        ax.plot(spectrum1.wavelength, spectrum1.flux,
                lw=self.linewidth,
                color=self.colormap,
                label=labels[0])

        # Segundo espectro
        ax.plot(spectrum2.wavelength, spectrum2.flux,
                lw=self.linewidth,
                color='red',
                linestyle='--',
                alpha=0.7,
                label=labels[1])

        # Configuración
        self._configure_axes(ax,
                             spectrum1.wavelength,
                             spectrum1.flux)

        ax.legend()
        return fig

    def present_filter_response(self, filter_data, spectrum_data=None):
        """Muestra la respuesta de un filtro"""
        fig, ax = self._create_figure()

        # Respuesta del filtro
        filter_line = ax.plot(filter_data.wavelength,
                              filter_data.response,
                              color='green',
                              alpha=0.7,
                              label='Respuesta del filtro')[0]

        # Espectro normalizado si está presente
        if spectrum_data is not None:
            norm_spectrum = self._normalize_spectrum(spectrum_data)
            spec_line = ax.plot(spectrum_data.wavelength,
                                norm_spectrum,
                                color=self.colormap,
                                alpha=0.5,
                                label='Espectro normalizado')[0]

            # Rellenar área bajo el filtro
            self._fill_filter_area(ax, filter_data, norm_spectrum)

        ax.legend()
        return fig

    def format_spectrum_metadata(self, spectrum_data):
        """Formatea metadatos para visualización"""
        meta = spectrum_data.meta.copy()
        formatted = [
            f"Instrumento: {meta.get('instrument', 'Desconocido')}",
            f"Observatorio: {meta.get('observatory', 'Desconocido')}",
            f"Fecha: {meta.get('date-obs', 'No disponible')}",
            f"Tiempo exposición: {meta.get('exptime', 'N/A')} s"
        ]
        return '\n'.join(formatted)

    # Métodos auxiliares
    def _create_figure(self):
        """Inicializa figura y ejes"""
        if self._current_figure:
            plt.close(self._current_figure)

        self._current_figure, self._current_axes = plt.subplots(
            figsize=self.figsize)
        return self._current_figure, self._current_axes

    def _plot_errors(self, ax, wavelength, flux, error):
        """Dibuja regiones de error"""
        ax.fill_between(wavelength,
                        flux - error,
                        flux + error,
                        color=self.colormap,
                        alpha=0.3)

    def _configure_axes(self, ax, wavelength, flux,
                        w_range=None, f_range=None):
        """Configura límites y etiquetas de ejes"""
        ax.set_xlabel(r'Longitud de onda ($\AA$)')
        ax.set_ylabel(r'Flujo ($erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$)')

        # Límites de ejes
        w_min, w_max = w_range or (wavelength.min(), wavelength.max())
        f_min, f_max = f_range or self._smart_flux_limits(flux)

        ax.set_xlim(w_min, w_max)
        ax.set_ylim(f_min, f_max)

        # Rejilla
        ax.grid(True, alpha=0.3)

    def _smart_flux_limits(self, flux, percentile=5):
        """Calcula límites inteligentes para el flujo"""
        valid_flux = flux[np.isfinite(flux)]
        if valid_flux.size == 0:
            return 0, 1

        p_low = np.percentile(valid_flux, percentile)
        p_high = np.percentile(valid_flux, 100 - percentile)
        return p_low, p_high

    def _normalize_spectrum(self, spectrum_data):
        """Normaliza el espectro para superposición con filtro"""
        flux = spectrum_data.flux
        return (flux - flux.min()) / (flux.max() - flux.min())

    def _fill_filter_area(self, ax, filter_data, norm_spectrum):
        """Rellena el área bajo el filtro"""
        interp_response = np.interp(filter_data.wavelength,
                                    filter_data.wavelength,
                                    filter_data.response)

        ax.fill_between(filter_data.wavelength,
                        norm_spectrum * interp_response,
                        color='green',
                        alpha=0.2)

    def _add_metadata_annotations(self, ax, metadata):
        """Añade metadatos como anotaciones"""
        text = self.format_spectrum_metadata(metadata)
        ax.annotate(text, xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round',
                              facecolor='white',
                              alpha=0.8))