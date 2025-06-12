"""Controlador principal del sistema CubeViewer."""

import numpy as np
import os
from typing import Optional, Tuple, List
from viewcube.data import DataManager, SpectrumData
from viewcube.data.filter_manager import FilterManager
from viewcube.ui.views import SpaxelViewer
from viewcube.ui.views import SpectrumViewer
from viewcube.utils.geometry_utils import GeometryUtils


class CubeController:
    """Controlador principal que coordina todos los componentes."""

    def __init__(self, name_fits: str, **kwargs):
        # Inicializar gestores de datos
        self.data_manager = DataManager()
        self.filter_manager = FilterManager(kwargs.get('dfilter', 'filters/'))

        # Inicializar visualizadores
        self.spaxel_viewer = SpaxelViewer(kwargs.get('fig_spaxel_size', (7.1, 6)))
        self.spectrum_viewer = SpectrumViewer(kwargs.get('fig_spectra_size', (8, 5)))

        # Utilidades
        self.geometry_utils = GeometryUtils()

        # Estado de la aplicación
        self.name_fits = name_fits
        self.current_spaxel: Optional[Tuple[int, int]] = None
        self.selected_spaxels: List[Tuple[int, int]] = []

        # Configuración desde kwargs
        self._setup_configuration(kwargs)

        # Cargar datos
        self._load_data(kwargs)

        # Configurar visualizadores
        self._setup_viewers()

        # Conectar eventos
        self._connect_events()

    def _setup_configuration(self, kwargs: dict) -> None:
        """Configura parámetros desde kwargs."""
        self.multiplicative_factor = kwargs.get('fo', 1.0)
        self.comparison_factor = kwargs.get('fc', 1.0)
        self.colorbar_enabled = kwargs.get('colorbar', True)
        self.default_filter = kwargs.get('default_filter', 'Halpha_KPNO-NOAO')
        self.norm_function = kwargs.get('norm', 'sqrt')
        self.wavelength_limits = kwargs.get('wlim', None)
        self.flux_limits = kwargs.get('flim', None)

    def _load_data(self, kwargs: dict) -> None:
        """Carga datos principales y de comparación."""
        # Cargar datos principales
        self.data_manager.load_primary_data(self.name_fits, **kwargs)

        # Cargar datos de comparación si existen
        comparison_file = kwargs.get('fitscom')
        if comparison_file:
            self.data_manager.load_comparison_data(comparison_file, **kwargs)

        # Aplicar factores multiplicativos
        self.data_manager.apply_multiplicative_factors(
            self.multiplicative_factor, self.comparison_factor
        )

    def _setup_viewers(self) -> None:
        """Configura visualizadores iniciales."""
        # Configurar filtro por defecto
        self.filter_manager.set_default_filter(self.default_filter)

        # Configurar visualizador de spaxels
        self._update_spaxel_display()

        if self.colorbar_enabled:
            self.spaxel_viewer.add_colorbar()

        # Configurar límites del visualizador de espectros
        if self.wavelength_limits:
            wl_min, wl_max = self.wavelength_limits
            self.spectrum_viewer.set_wavelength_limits(wl_min, wl_max)
        else:
            wl_min, wl_max = self.data_manager.get_wavelength_limits()
            # Añadir padding del 5%
            padding = (wl_max - wl_min) * 0.05
            self.spectrum_viewer.set_wavelength_limits(wl_min - padding, wl_max + padding)

        if self.flux_limits:
            flux_min, flux_max = self.flux_limits
            self.spectrum_viewer.set_flux_limits(flux_min, flux_max)

    def _update_spaxel_display(self) -> None:
        """Actualiza visualización de spaxels."""
        primary_data = self.data_manager.primary_data
        if not primary_data:
            return

        # Calcular datos de color usando filtro actual
        color_data = self._calculate_color_data()

        # Configurar extent basado en geometría
        if primary_data.ndim == 3:
            extent = self._calculate_extent(primary_data)
            self.spaxel_viewer.set_image_data(color_data, extent)

        self.spaxel_viewer.set_title(os.path.basename(self.name_fits))

    def _calculate_color_data(self) -> np.ndarray:
        """Calcula datos de color aplicando filtro actual."""
        primary_data = self.data_manager.primary_data

        if primary_data.ndim == 2:
            # Para datos RSS, sumar sobre longitudes de onda
            return np.sum(primary_data.data, axis=1)
        else:
            # Para cubos 3D, aplicar filtro si está disponible
            filter_data = self.filter_manager.load_filter_data()
            if filter_data:
                filtered_flux = self.filter_manager.apply_filter_to_spectrum(
                    primary_data.wavelength, primary_data.data
                )
                return filtered_flux
            else:
                # Suma simple sobre longitudes de onda
                return np.sum(primary_data.data, axis=0)

    def _calculate_extent(self, data: SpectrumData) -> List[float]:
        """Calcula extent para visualización de cubos 3D."""
        if hasattr(data, 'fobj') and hasattr(data.fobj, 'hdr'):
            header = data.fobj.hdr
            try:
                cdelt1 = header.get('CDELT1', 1.0)
                cdelt2 = header.get('CDELT2', 1.0)
                crpix1 = header.get('CRPIX1', data.x_size // 2)
                crpix2 = header.get('CRPIX2', data.y_size // 2)

                # Calcular resolución espacial
                sr = cdelt1 if cdelt1 != 0 else 1.0

                extent = [
                    -crpix1 * sr,
                    (data.x_size - crpix1) * sr,
                    -crpix2 * sr,
                    (data.y_size - crpix2) * sr
                ]
                return extent
            except:
                pass

        # Extent por defecto
        return [0, data.x_size if hasattr(data, 'x_size') else data.shape[-1],
                0, data.y_size if hasattr(data, 'y_size') else data.shape[-2]]

    def _connect_events(self) -> None:
        """Conecta eventos entre componentes."""
        self.spaxel_viewer.set_click_callback(self._on_spaxel_click)
        self.spaxel_viewer.set_motion_callback(self._on_spaxel_motion)
        self.spectrum_viewer.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.spaxel_viewer.figure.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_spaxel_click(self, event) -> None:
        """Maneja clicks en el visualizador de spaxels."""
        if not event.inaxes:
            return

        # Convertir coordenadas del click a índices de spaxel
        x, y = event.xdata, event.ydata
        spaxel_coords = self._screen_to_spaxel_coordinates(x, y)

        if spaxel_coords:
            if event.button == 1:  # Click izquierdo
                self._select_spaxel(spaxel_coords)
            elif event.button == 3:  # Click derecho
                self._show_multiple_spectra()

    def _on_spaxel_motion(self, event) -> None:
        """Maneja movimiento del mouse sobre spaxels."""
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        spaxel_coords = self._screen_to_spaxel_coordinates(x, y)

        if spaxel_coords:
            self._update_spectrum_display(spaxel_coords)

    def _screen_to_spaxel_coordinates(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convierte coordenadas de pantalla a coordenadas de spaxel."""
        primary_data = self.data_manager.primary_data
        if not primary_data or primary_data.ndim != 3:
            return None

        # Conversión básica - puede necesitar refinamiento según la geometría específica
        spaxel_x = int(x) if 0 <= x < primary_data.x_size else None
        spaxel_y = int(y) if 0 <= y < primary_data.y_size else None

        return (spaxel_x, spaxel_y) if spaxel_x is not None and spaxel_y is not None else None

    def _select_spaxel(self, coords: Tuple[int, int]) -> None:
        """Selecciona un spaxel específico."""
        x, y = coords

        if coords in self.selected_spaxels:
            # Deseleccionar
            self.selected_spaxels.remove(coords)
            self.spaxel_viewer.remove_selection(len(self.selected_spaxels))
        else:
            # Seleccionar
            self.selected_spaxels.append(coords)
            self.spaxel_viewer.add_selection(
                len(self.selected_spaxels) - 1,
                (x, y),
                radius=0.5
            )

        self._update_spectrum_display(coords)

    def _update_spectrum_display(self, coords: Tuple[int, int]) -> None:
        """Actualiza visualización de espectro para coordenadas dadas."""
        x, y = coords
        primary_data = self.data_manager.primary_data

        if not primary_data:
            return

        self.current_spaxel = coords

        # Obtener espectro
        if primary_data.ndim == 3:
            spectrum = primary_data.get_spectrum_3d(x, y)
        else:
            # Para datos RSS, usar índice lineal
            index = y * primary_data.shape[1] + x if x < primary_data.shape[1] else 0
            spectrum = primary_data.get_spectrum(index)

        # Plotear espectro principal
        self.spectrum_viewer.plot_spectrum(
            primary_data.wavelength,
            spectrum,
            label=f"Spaxel ({x}, {y})"
        )

        # Plotear comparación si existe
        comparison_data = self.data_manager.comparison_data
        if comparison_data:
            if comparison_data.ndim == 3:
                comp_spectrum = comparison_data.get_spectrum_3d(x, y)
            else:
                comp_spectrum = comparison_data.get_spectrum(index)

            self.spectrum_viewer.plot_comparison(
                comparison_data.wavelength,
                comp_spectrum,
                label="Comparison"
            )

        # Añadir filtro si está disponible
        filter_data = self.filter_manager.load_filter_data()
        if filter_data:
            filter_wl, filter_trans = filter_data
            scale_factor = np.max(spectrum) * 1.2  # Factor de escala para visualización
            interpolated_filter = self.filter_manager.interpolate_filter(primary_data.wavelength)
            self.spectrum_viewer.plot_filter(
                primary_data.wavelength,
                interpolated_filter,
                scale_factor=scale_factor
            )

    def _show_multiple_spectra(self) -> None:
        """Muestra múltiples espectros seleccionados."""
        if not self.selected_spaxels:
            return

        self.spectrum_viewer.clear_display()

        # Plotear espectros individuales
        for i, (x, y) in enumerate(self.selected_spaxels):
            primary_data = self.data_manager.primary_data
            if primary_data.ndim == 3:
                spectrum = primary_data.get_spectrum_3d(x, y)
            else:
                index = y * primary_data.shape[1] + x
                spectrum = primary_data.get_spectrum(index)

            self.spectrum_viewer.axes.plot(
                primary_data.wavelength,
                spectrum,
                label=f"Spaxel {i + 1} ({x}, {y})",
                picker=True
            )

        # Plotear espectro integrado
        if len(self.selected_spaxels) > 1:
            integrated_spectrum = self._calculate_integrated_spectrum()
            self.spectrum_viewer.axes.plot(
                primary_data.wavelength,
                integrated_spectrum,
                label="Integrated",
                linewidth=2,
                picker=True
            )

        self.spectrum_viewer.add_legend()

    def _calculate_integrated_spectrum(self) -> np.ndarray:
        """Calcula espectro integrado de spaxels seleccionados."""
        primary_data = self.data_manager.primary_data
        integrated = np.zeros(len(primary_data.wavelength))

        for x, y in self.selected_spaxels:
            if primary_data.ndim == 3:
                spectrum = primary_data.get_spectrum_3d(x, y)
            else:
                index = y * primary_data.shape[1] + x
                spectrum = primary_data.get_spectrum(index)

            integrated += spectrum

        return integrated

    def _on_key_press(self, event) -> None:
        """Maneja eventos de teclado."""
        if event.key == "t":
            # Cambiar al siguiente filtro
            self.filter_manager.next_filter()
            self._update_spaxel_display()
            if self.current_spaxel:
                self._update_spectrum_display(self.current_spaxel)

        elif event.key == "T":
            # Cambiar al filtro anterior
            self.filter_manager.previous_filter()
            self._update_spaxel_display()
            if self.current_spaxel:
                self._update_spectrum_display(self.current_spaxel)

        elif event.key == "*":
            # Limpiar selecciones
            self.spaxel_viewer.clear_selections()
            self.spectrum_viewer.clear_display()
            self.selected_spaxels.clear()

        elif event.key == "s":
            # Cambiar modo de visualización
            if len(self.selected_spaxels) > 0:
                self._show_multiple_spectra()

        elif event.key == "q":
            # Salir
            import sys
            sys.exit()

    def save_spectra(self, filename_base: str, save_fits: bool = False,
                     save_txt: bool = True, save_integrated: bool = True,
                     save_individual: bool = False) -> None:
        """Guarda espectros seleccionados."""
        if not self.selected_spaxels:
            print("No spectra selected to save")
            return

        primary_data = self.data_manager.primary_data

        if save_integrated and len(self.selected_spaxels) > 1:
            integrated_spectrum = self._calculate_integrated_spectrum()
            self._save_single_spectrum(
                primary_data.wavelength,
                integrated_spectrum,
                f"{filename_base}_integrated",
                save_fits,
                save_txt,
                info=f"Integrated spectrum from {len(self.selected_spaxels)} spaxels"
            )

        if save_individual:
            for i, (x, y) in enumerate(self.selected_spaxels):
                if primary_data.ndim == 3:
                    spectrum = primary_data.get_spectrum_3d(x, y)
                else:
                    index = y * primary_data.shape[1] + x
                    spectrum = primary_data.get_spectrum(index)

                self._save_single_spectrum(
                    primary_data.wavelength,
                    spectrum,
                    f"{filename_base}_spaxel_{x:03d}_{y:03d}",
                    save_fits,
                    save_txt,
                    info=f"Spectrum from spaxel ({x}, {y})"
                )

    def _save_single_spectrum(self, wavelength: np.ndarray, spectrum: np.ndarray,
                              filename: str, save_fits: bool, save_txt: bool,
                              info: str = "") -> None:
        """Guarda un espectro individual."""
        from viewcube.utils import save_spec

        header_info = None
        if hasattr(self.data_manager.primary_data, 'header'):
            header_info = self.data_manager.primary_data.header

        save_spec(
            wavelength,
            spectrum,
            filename,
            fits=save_fits,
            txt=save_txt,
            hd=header_info,
            infotxt=[info, f"Extracted from: {self.name_fits}"]
        )
