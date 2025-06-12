# core/controllers/main_controller.py
"""Controlador principal que unifica toda la funcionalidad."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod

from ..models.spectrum_data import SpectrumData
from ..services.data_service import DataService
from ..services.filter_service import FilterService
from viewcube.core.services.sonification_service import SonificationService
from ...ui.viewers.spaxel_viewer import SpaxelViewer
from ...ui.viewers.spectrum_viewer import SpectrumViewer
from ...ui.viewers.rss_viewer import RSSViewer
from viewcube.ui.dialogs.window_manager import WindowManager
from ...ui.dialogs.limits_dialog import LimitsDialog
from ...utils.event_manager import EventManager


class ViewerController(ABC):
    """Controlador base abstracto."""

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def setup_ui(self) -> None:
        pass


class MainController(ViewerController):
    """Controlador principal que mantiene toda la funcionalidad original."""

    def __init__(self, filename: str, **kwargs):
        # Servicios de datos
        self.data_service = DataService()
        self.filter_service = FilterService(kwargs.get('dfilter', 'filters/'))
        self.sonification_service = SonificationService() if kwargs.get('soni_start') else None

        # Visualizadores
        self.spaxel_viewer = SpaxelViewer(kwargs.get('fig_spaxel_size', (7.1, 6)))
        self.spectrum_viewer = SpectrumViewer(kwargs.get('fig_spectra_size', (8, 5)))
        self.rss_viewer = None

        # Gestores de UI
        self.window_manager = WindowManager(kwargs.get('fig_window_manager', (5, 5)))
        self.limits_dialog = LimitsDialog()

        # Gestor de eventos
        self.event_manager = EventManager(self)

        # Estado de la aplicación
        self.filename = filename
        self.config = kwargs
        self.current_spaxel: Optional[Tuple[int, int]] = None
        self.selected_spaxels: List[Tuple[int, int]] = []

        # Datos
        self.primary_data: Optional[SpectrumData] = None
        self.comparison_data: Optional[SpectrumData] = None

        # Inicialización
        self._load_data()
        self.setup_ui()
        self._connect_events()

    def _load_data(self) -> None:
        """Carga datos principales y de comparación."""
        # Cargar datos principales
        self.primary_data = self.data_service.load_primary_data(
            self.filename, **self.config
        )

        # Cargar datos de comparación si existen
        comparison_file = self.config.get('fitscom')
        if comparison_file:
            self.comparison_data = self.data_service.load_comparison_data(
                comparison_file, **self.config
            )

        # Determinar tipo de visor según datos
        if self.primary_data.is_rss_data() or self.config.get('ptable'):
            self._setup_rss_viewer()

    def _setup_rss_viewer(self) -> None:
        """Configura visor RSS para datos específicos."""
        self.rss_viewer = RSSViewer(
            self.filename,
            self.config.get('ptable'),
            **self.config
        )

    def setup_ui(self) -> None:
        """Configura interfaz de usuario completa."""
        # Configurar filtro por defecto
        default_filter = self.config.get('default_filter', 'Halpha_KPNO-NOAO')
        self.filter_service.set_default_filter(default_filter)

        # Configurar visualizadores
        self._setup_spaxel_display()
        self._setup_spectrum_display()

        # Configurar sonificación si está habilitada
        if self.sonification_service and self.config.get('soni_start'):
            self.sonification_service.initialize(
                self.spaxel_viewer.figure,
                file=self.filename,
                data=self.primary_data.data if self.primary_data else None,
                **self.config
            )

    def _setup_spaxel_display(self) -> None:
        """Configura visualización de spaxels."""
        if not self.primary_data:
            return

        # Calcular datos de color usando filtro actual
        color_data = self._calculate_color_data()

        # Configurar extent
        extent = self._calculate_extent()

        # Actualizar visualizador
        self.spaxel_viewer.set_image_data(color_data, extent)
        self.spaxel_viewer.set_title(f"Spaxel Viewer - {self.filename}")

        # Añadir colorbar si está habilitado
        if self.config.get('colorbar', True):
            self.spaxel_viewer.add_colorbar()

    def _calculate_color_data(self) -> np.ndarray:
        """Calcula datos de color aplicando filtro actual."""
        if not self.primary_data:
            return np.zeros((10, 10))

        data = self.primary_data.data
        wavelength = self.primary_data.wavelength

        # Aplicar filtro si está disponible
        filter_data = self.filter_service.get_current_filter_data()
        if filter_data:
            return self.filter_service.apply_filter_to_spectrum(wavelength, data)

        # Suma simple sobre longitudes de onda
        if data.ndim == 3:
            return np.nansum(data, axis=0)
        else:
            return np.nansum(data, axis=1)

    def _calculate_extent(self) -> List[float]:
        """Calcula extent para visualización."""
        if not self.primary_data or not hasattr(self.primary_data, 'fobj'):
            return [0, 10, 0, 10]

        header = self.primary_data.fobj.hdr
        try:
            cdelt1 = header.get('CDELT1', 1.0)
            cdelt2 = header.get('CDELT2', 1.0)
            crpix1 = header.get('CRPIX1', self.primary_data.x_size // 2)
            crpix2 = header.get('CRPIX2', self.primary_data.y_size // 2)

            sr = cdelt1 if cdelt1 != 0 else 1.0
            return [
                -crpix1 * sr,
                (self.primary_data.x_size - crpix1) * sr,
                -crpix2 * sr,
                (self.primary_data.y_size - crpix2) * sr
            ]
        except:
            return [0, self.primary_data.x_size, 0, self.primary_data.y_size]

    def _setup_spectrum_display(self) -> None:
        """Configura visualización de espectros."""
        # Configurar límites de longitud de onda
        wlim = self.config.get('wlim')
        if wlim:
            self.spectrum_viewer.set_wavelength_limits(wlim[0], wlim[1])
        elif self.primary_data:
            wl_min, wl_max = np.min(self.primary_data.wavelength), np.max(self.primary_data.wavelength)
            padding = (wl_max - wl_min) * 0.05
            self.spectrum_viewer.set_wavelength_limits(wl_min - padding, wl_max + padding)

        # Configurar límites de flujo
        flim = self.config.get('flim')
        if flim:
            self.spectrum_viewer.set_flux_limits(flim[0], flim[1])

    def _connect_events(self) -> None:
        """Conecta todos los eventos del sistema."""
        self.event_manager.connect_all_events()

    def on_spaxel_click(self, event) -> None:
        """Maneja clicks en spaxels."""
        if not event.inaxes or not self.primary_data:
            return

        x, y = event.xdata, event.ydata
        spaxel_coords = self._screen_to_spaxel_coordinates(x, y)

        if spaxel_coords:
            if event.button == 1:  # Click izquierdo
                self._select_spaxel(spaxel_coords)
            elif event.button == 3:  # Click derecho
                self._show_multiple_spectra()

    def on_spaxel_motion(self, event) -> None:
        """Maneja movimiento del mouse sobre spaxels."""
        if not event.inaxes or not self.primary_data:
            return

        x, y = event.xdata, event.ydata
        spaxel_coords = self._screen_to_spaxel_coordinates(x, y)

        if spaxel_coords:
            self._update_spectrum_display(spaxel_coords)

            # Sonificación en tiempo real
            if self.sonification_service:
                spectrum = self._get_spectrum_at_coordinates(spaxel_coords)
                if spectrum is not None:
                    self.sonification_service.sonify(*spaxel_coords, spectrum)

    def _screen_to_spaxel_coordinates(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convierte coordenadas de pantalla a spaxel."""
        if not self.primary_data:
            return None

        # Conversión básica - puede necesitar refinamiento
        spaxel_x = int(round(x))
        spaxel_y = int(round(y))

        # Verificar límites
        if (0 <= spaxel_x < self.primary_data.x_size and
                0 <= spaxel_y < self.primary_data.y_size):
            return (spaxel_x, spaxel_y)

        return None

    def _select_spaxel(self, coords: Tuple[int, int]) -> None:
        """Selecciona un spaxel."""
        if coords in self.selected_spaxels:
            # Deseleccionar
            self.selected_spaxels.remove(coords)
            self.spaxel_viewer.remove_selection(coords)
        else:
            # Seleccionar
            self.selected_spaxels.append(coords)
            self.spaxel_viewer.add_selection(coords)

        self._update_spectrum_display(coords)

    def _update_spectrum_display(self, coords: Tuple[int, int]) -> None:
        """Actualiza visualización de espectro."""
        self.current_spaxel = coords

        # Obtener espectro principal
        spectrum = self._get_spectrum_at_coordinates(coords)
        if spectrum is None:
            return

        # Plotear espectro principal
        self.spectrum_viewer.plot_spectrum(
            self.primary_data.wavelength,
            spectrum,
            label=f"Spaxel ({coords[0]}, {coords[1]})"
        )

        # Plotear comparación si existe
        if self.comparison_data:
            comp_spectrum = self._get_comparison_spectrum_at_coordinates(coords)
            if comp_spectrum is not None:
                self.spectrum_viewer.plot_comparison(
                    self.comparison_data.wavelength,
                    comp_spectrum,
                    label="Comparison"
                )

        # Añadir filtro si está disponible
        filter_data = self.filter_service.get_current_filter_data()
        if filter_data:
            filter_wl, filter_trans = filter_data
            scale_factor = np.max(spectrum) * 1.2
            interpolated_filter = np.interp(
                self.primary_data.wavelength, filter_wl, filter_trans
            )
            self.spectrum_viewer.plot_filter(
                self.primary_data.wavelength,
                interpolated_filter,
                scale_factor=scale_factor
            )

    def _get_spectrum_at_coordinates(self, coords: Tuple[int, int]) -> Optional[np.ndarray]:
        """Obtiene espectro en coordenadas específicas."""
        if not self.primary_data:
            return None

        x, y = coords
        try:
            if self.primary_data.data.ndim == 3:
                return self.primary_data.data[:, y, x]
            else:
                # Para datos RSS
                index = y * self.primary_data.data.shape[1] + x
                if 0 <= index < self.primary_data.data.shape[0]:
                    return self.primary_data.data[index, :]
        except IndexError:
            pass

        return None

    def _get_comparison_spectrum_at_coordinates(self, coords: Tuple[int, int]) -> Optional[np.ndarray]:
        """Obtiene espectro de comparación en coordenadas específicas."""
        if not self.comparison_data:
            return None

        x, y = coords
        try:
            if self.comparison_data.data.ndim == 3:
                return self.comparison_data.data[:, y, x]
            else:
                index = y * self.comparison_data.data.shape[1] + x
                if 0 <= index < self.comparison_data.data.shape[0]:
                    return self.comparison_data.data[index, :]
        except IndexError:
            pass

        return None

    def _show_multiple_spectra(self) -> None:
        """Muestra múltiples espectros seleccionados."""
        if not self.selected_spaxels:
            return

        self.spectrum_viewer.clear_display()

        # Plotear espectros individuales
        for i, coords in enumerate(self.selected_spaxels):
            spectrum = self._get_spectrum_at_coordinates(coords)
            if spectrum is not None:
                self.spectrum_viewer.axes.plot(
                    self.primary_data.wavelength,
                    spectrum,
                    label=f"Spaxel {i + 1} ({coords[0]}, {coords[1]})",
                    picker=True
                )

        # Plotear espectro integrado
        if len(self.selected_spaxels) > 1:
            integrated = self._calculate_integrated_spectrum()
            self.spectrum_viewer.axes.plot(
                self.primary_data.wavelength,
                integrated,
                label="Integrated",
                linewidth=2,
                picker=True
            )

        self.spectrum_viewer.add_legend()

    def _calculate_integrated_spectrum(self) -> np.ndarray:
        """Calcula espectro integrado."""
        if not self.primary_data:
            return np.zeros(100)

        integrated = np.zeros(len(self.primary_data.wavelength))
        for coords in self.selected_spaxels:
            spectrum = self._get_spectrum_at_coordinates(coords)
            if spectrum is not None:
                integrated += spectrum

        return integrated

    # Métodos de eventos del teclado
    def next_filter(self) -> None:
        """Cambia al siguiente filtro."""
        self.filter_service.next_filter()
        self._setup_spaxel_display()
        if self.current_spaxel:
            self._update_spectrum_display(self.current_spaxel)

    def previous_filter(self) -> None:
        """Cambia al filtro anterior."""
        self.filter_service.previous_filter()
        self._setup_spaxel_display()
        if self.current_spaxel:
            self._update_spectrum_display(self.current_spaxel)

    def clear_selections(self) -> None:
        """Limpia todas las selecciones."""
        self.spaxel_viewer.clear_selections()
        self.spectrum_viewer.clear_display()
        self.selected_spaxels.clear()
        self.current_spaxel = None

    def show_window_manager(self) -> None:
        """Muestra gestor de ventanas."""
        self.window_manager.show(self)

    def show_lambda_limits(self) -> None:
        """Muestra diálogo de límites de longitud de onda."""
        current_limits = self.spectrum_viewer.get_wavelength_limits()
        new_limits = self.limits_dialog.get_wavelength_limits(current_limits)
        if new_limits:
            self.spectrum_viewer.set_wavelength_limits(*new_limits)

    def show_flux_limits(self) -> None:
        """Muestra diálogo de límites de flujo."""
        current_limits = self.spectrum_viewer.get_flux_limits()
        new_limits = self.limits_dialog.get_flux_limits(current_limits)
        if new_limits:
            self.spectrum_viewer.set_flux_limits(*new_limits)

    def save_spectra(self, filename_base: str = None) -> None:
        """Guarda espectros seleccionados."""
        if not self.selected_spaxels:
            print("No spectra selected to save")
            return

        if not filename_base:
            filename_base = self.filename.replace('.fits', '_extracted')

        # Configuración de guardado
        save_fits = self.config.get('fits', False)
        save_txt = self.config.get('txt', True)
        save_integrated = self.config.get('integrated', True)
        save_individual = self.config.get('individual', False)

        # Guardar espectro integrado
        if save_integrated and len(self.selected_spaxels) > 1:
            integrated = self._calculate_integrated_spectrum()
            self._save_single_spectrum(
                self.primary_data.wavelength,
                integrated,
                f"{filename_base}_integrated",
                save_fits,
                save_txt,
                info=f"Integrated spectrum from {len(self.selected_spaxels)} spaxels"
            )

        # Guardar espectros individuales
        if save_individual:
            for i, coords in enumerate(self.selected_spaxels):
                spectrum = self._get_spectrum_at_coordinates(coords)
                if spectrum is not None:
                    self._save_single_spectrum(
                        self.primary_data.wavelength,
                        spectrum,
                        f"{filename_base}_spaxel_{coords[0]:03d}_{coords[1]:03d}",
                        save_fits,
                        save_txt,
                        info=f"Spectrum from spaxel ({coords[0]}, {coords[1]})"
                    )

    def _save_single_spectrum(self, wavelength: np.ndarray, spectrum: np.ndarray,
                              filename: str, save_fits: bool, save_txt: bool,
                              info: str = "") -> None:
        """Guarda un espectro individual."""
        from ...utils.io_utils import save_spectrum

        save_spectrum(
            wavelength,
            spectrum,
            filename,
            fits=save_fits,
            txt=save_txt,
            header=getattr(self.primary_data, 'header', None),
            info=[info, f"Extracted from: {self.filename}"]
        )

    def run(self) -> None:
        """Ejecuta el controlador principal."""
        # Si hay datos RSS, usar visor RSS
        if self.rss_viewer:
            self.rss_viewer.run()
        else:
            # Mostrar ventanas principales
            plt.show()
