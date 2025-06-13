import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ...core.interfaces.presenter_interfaces import CubePresenterInterface
from ..repositories.fits_repository import FitsRepository


class CubePresenter(CubePresenterInterface):
    """Presentador para visualización de cubos de datos 3D"""

    def __init__(self, colormap='viridis', scale='linear', figsize=(10, 8)):
        self.colormap = colormap
        self.scale = scale
        self.figsize = figsize
        self._current_figure = None
        self._current_axes = None
        self._colorbar = None

    def present_slice(self, cube_data, wavelength_index,
                      colormap=None, scale=None):
        """Visualiza un corte 2D del cubo"""
        fig, ax = self._create_figure()

        # Obtener datos
        data_slice = cube_data.data[wavelength_index]
        wavelength = cube_data.wavelength[wavelength_index]

        # Configurar normalización
        norm = self._get_norm(data_slice, scale)

        # Mostrar imagen
        im = ax.imshow(data_slice,
                       cmap=colormap or self.colormap,
                       origin='lower',
                       norm=norm)

        # Configuraciones
        self._configure_axes(ax, wavelength)
        self._add_colorbar(fig, ax, im)

        return fig

    def present_integrated_map(self, cube_data, wavelength_range=None,
                               colormap=None, scale=None):
        """Mapa integrado en un rango espectral"""
        fig, ax = self._create_figure()

        # Determinar índices del rango
        start_idx, end_idx = self._get_wave_indices(cube_data, wavelength_range)

        # Integrar datos
        integrated_data = np.nanmean(cube_data.data[start_idx:end_idx], axis=0)

        # Configurar normalización
        norm = self._get_norm(integrated_data, scale)

        # Mostrar imagen
        im = ax.imshow(integrated_data,
                       cmap=colormap or self.colormap,
                       origin='lower',
                       norm=norm)

        # Configuraciones
        self._configure_axes(ax,
                             f"Integrado {cube_data.wavelength[start_idx]:.1f}-"
                             f"{cube_data.wavelength[end_idx]:.1f} Å")
        self._add_colorbar(fig, ax, im)

        return fig

    def present_spaxel_grid(self, cube_data, positions,
                            wavelength_range=None):
        """Muestra una cuadrícula de espectros"""
        fig, axes = self._create_grid_figure(len(positions))

        # Determinar rango espectral
        start_idx, end_idx = self._get_wave_indices(cube_data, wavelength_range)

        for ax, (x, y) in zip(axes.flatten(), positions):
            if x >= cube_data.nx or y >= cube_data.ny:
                continue

            spectrum = cube_data.data[:, y, x]
            wavelength = cube_data.wavelength

            ax.plot(wavelength[start_idx:end_idx],
                    spectrum[start_idx:end_idx],
                    lw=1,
                    color=self.colormap)

            ax.set_title(f"Spaxel ({x}, {y})")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def format_cube_metadata(self, cube_data):
        """Formatea metadatos del cubo"""
        meta = cube_data.meta.copy()
        formatted = [
            f"Dimensión: {cube_data.data.shape}",
            f"Rango λ: {cube_data.wavelength[0]:.1f}-{cube_data.wavelength[-1]:.1f} Å",
            f"Instrumento: {meta.get('instrument', 'Desconocido')}",
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

    def _create_grid_figure(self, n_spectra):
        """Crea una cuadrícula de subplots"""
        grid_size = int(np.ceil(np.sqrt(n_spectra)))
        fig, axes = plt.subplots(grid_size, grid_size,
                                 figsize=(self.figsize[0],
                                          self.figsize[0]))
        return fig, axes

    def _get_norm(self, data, scale=None):
        """Obtiene el objeto de normalización"""
        scale = scale or self.scale
        vmin, vmax = np.nanmin(data), np.nanmax(data)

        if scale == 'log':
            return LogNorm(vmin=vmin, vmax=vmax)
        if scale == 'power':
            return PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        return None

    def _configure_axes(self, ax, title):
        """Configura ejes y título"""
        ax.set_xlabel('X (píxeles)')
        ax.set_ylabel('Y (píxeles)')
        ax.set_title(title)

    def _add_colorbar(self, fig, ax, im):
        """Añade barra de colores"""
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        self._colorbar = fig.colorbar(im, cax=cax)
        self._colorbar.set_label(r'Flujo ($erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$)')

    def _get_wave_indices(self, cube_data, wavelength_range):
        """Calcula índices para un rango espectral"""
        if wavelength_range is None:
            return 0, len(cube_data.wavelength)

        wave = cube_data.wavelength
        start_idx = np.argmin(np.abs(wave - wavelength_range[0]))
        end_idx = np.argmin(np.abs(wave - wavelength_range[1]))
        return start_idx, end_idx