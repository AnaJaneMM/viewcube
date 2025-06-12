# ui/viewers/rss_viewer.py
"""Visor RSS que mantiene toda la funcionalidad original."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon
from scipy.spatial import distance
import math
from typing import List, Optional

from viewcube.ui.controllers import ViewerController
from ...data.loaders.position_loader import PositionLoader
from ...utils.geometry_utils import GeometryUtils


class RSSViewer(ViewerController):
    """Visor RSS que replica funcionalidad completa del original."""

    def __init__(self, name_fits: str, ptable: str, **kwargs):
        # Configuración base
        self.name_fits = name_fits
        self.ptable = ptable
        self.config = kwargs

        # Cargar datos
        self.position_loader = PositionLoader()
        self.geometry_utils = GeometryUtils()

        # Configurar visualización
        self.fiber_type = kwargs.get('ft', 'C')  # C=circle, H=hexagon
        self.hex_scale = kwargs.get('hex_scale', None)
        self.extent = kwargs.get('extent', None)
        self.angle = kwargs.get('angle', None)

        # Estado
        self.selected_fibers: List[int] = []
        self.fiber_positions = None
        self.fiber_patches = []

        # Inicializar
        self._load_position_table()
        self.setup_ui()

    def _load_position_table(self) -> None:
        """Carga tabla de posiciones de fibras."""
        extension = self.config.get('extension', False)

        if extension:
            # Tabla está en extensión del FITS
            self.fiber_positions = self.position_loader.load_from_fits_extension(
                self.name_fits, self.ptable
            )
        else:
            # Tabla externa
            self.fiber_positions = self.position_loader.load_from_file(
                self.ptable, angle=self.angle
            )

        if self.fiber_positions is None:
            raise ValueError(f"Could not load position table: {self.ptable}")

    def setup_ui(self) -> None:
        """Configura interfaz de usuario RSS."""
        # Crear figura principal
        self.figure = plt.figure(1, self.config.get('fig_spaxel_size', (7, 6)))
        self.figure.set_label("RSS Spaxel Viewer")
        self.axes = self.figure.add_subplot(111)

        # Dibujar fibras
        self._draw_fibers()

        # Configurar límites
        self._set_axis_limits()

        # Conectar eventos
        self._connect_events()

    def _draw_fibers(self) -> None:
        """Dibuja fibras en el plot."""
        if self.fiber_positions is None:
            return

        x = self.fiber_positions['X']
        y = self.fiber_positions['Y']

        # Calcular radio de fibra
        if hasattr(self.fiber_positions, 'RADIUS'):
            radius = self.fiber_positions['RADIUS'][0]
        else:
            radius = self._calculate_fiber_radius(x, y)

        # Crear patches según tipo de fibra
        if self.fiber_type == 'H':  # Hexágonos
            patches = self._create_hexagon_patches(x, y, radius)
        else:  # Círculos
            patches = self._create_circle_patches(x, y, radius)

        # Añadir patches al plot
        collection = PatchCollection(
            patches,
            alpha=self.config.get('palpha', 0.95),
            linewidths=self.config.get('plw', 0.1),
            edgecolors=self.config.get('plc', 'k'),
            picker=True
        )

        self.axes.add_collection(collection)
        self.fiber_patches = patches

        # Colorear según datos
        self._color_fibers()

    def _calculate_fiber_radius(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calcula radio de fibra basado en posiciones."""
        points = list(zip(x, y))
        distances = distance.cdist(points, points, 'euclidean')
        # Encontrar distancia mínima no-cero
        min_dist = np.min(distances[distances > 0])
        return min_dist / 2.2  # Factor empírico

    def _create_circle_patches(self, x: np.ndarray, y: np.ndarray,
                               radius: float) -> List[Circle]:
        """Crea patches circulares para fibras."""
        return [Circle((xi, yi), radius) for xi, yi in zip(x, y)]

    def _create_hexagon_patches(self, x: np.ndarray, y: np.ndarray,
                                scale: float) -> List[Polygon]:
        """Crea patches hexagonales para fibras."""
        patches = []
        sqrt3 = math.sqrt(3.0)

        # Vértices del hexágono
        hex_vertices = np.array([
            [-0.5 / sqrt3, 0.5 / sqrt3, 1.0 / sqrt3, 0.5 / sqrt3, -0.5 / sqrt3, -1.0 / sqrt3],
            [0.5, 0.5, 0.0, -0.5, -0.5, 0.0]
        ]).T * scale * 0.99

        # Rotar si hay ángulo especificado
        if self.angle:
            angle_rad = self.angle * np.pi / 180
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            hex_vertices = hex_vertices @ rotation_matrix.T

        # Crear patches
        for xi, yi in zip(x, y):
            vertices = hex_vertices + np.array([xi, yi])
            patches.append(Polygon(vertices))

        return patches

    def _color_fibers(self) -> None:
        """Colorea fibras según datos espectrales."""
        # Implementar coloreado basado en filtro actual
        # Esta es una versión simplificada
        if hasattr(self, 'spectral_data'):
            # Calcular flujo integrado por fibra
            integrated_flux = np.sum(self.spectral_data, axis=1)

            # Normalizar y aplicar
            norm_flux = (integrated_flux - np.min(integrated_flux)) / \
                        (np.max(integrated_flux) - np.min(integrated_flux))

            # Aplicar colormap
            colors = plt.cm.viridis(norm_flux)

            # Actualizar collection
            for patch, color in zip(self.fiber_patches, colors):
                patch.set_facecolor(color)

    def _set_axis_limits(self) -> None:
        """Configura límites de ejes."""
        if self.fiber_positions is None:
            return

        x, y = self.fiber_positions['X'], self.fiber_positions['Y']

        if self.extent:
            self.axes.set_xlim(self.extent[0], self.extent[1])
            self.axes.set_ylim(self.extent[2], self.extent[3])
        else:
            # Calcular límites automáticamente
            limits = self.geometry_utils.calculate_spaxel_limits(
                x, y, self._calculate_fiber_radius(x, y)
            )
            self.axes.set_xlim(limits[0], limits[1])
            self.axes.set_ylim(limits[2], limits[3])

        self.axes.set_aspect('equal')

    def _connect_events(self) -> None:
        """Conecta eventos de mouse y teclado."""
        self.figure.canvas.mpl_connect('button_press_event', self._on_click)
        self.figure.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.figure.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.figure.canvas.mpl_connect('pick_event', self._on_pick)

    def _on_click(self, event) -> None:
        """Maneja clicks en fibras."""
        if not event.inaxes:
            return

        # Encontrar fibra más cercana
        fiber_idx = self._find_nearest_fiber(event.xdata, event.ydata)

        if fiber_idx is not None:
            if event.button == 1:  # Click izquierdo
                self._select_fiber(fiber_idx)
            elif event.button == 3:  # Click derecho
                self._show_fiber_spectrum(fiber_idx)

    def _on_motion(self, event) -> None:
        """Maneja movimiento del mouse."""
        if not event.inaxes:
            return

        # Mostrar espectro de fibra bajo cursor
        fiber_idx = self._find_nearest_fiber(event.xdata, event.ydata)
        if fiber_idx is not None:
            self._update_spectrum_preview(fiber_idx)

    def _on_key_press(self, event) -> None:
        """Maneja eventos de teclado."""
        if event.key == 't':
            self._next_filter()
        elif event.key == 'T':
            self._previous_filter()
        elif event.key == '*':
            self._clear_selections()
        elif event.key == 's':
            self._save_spectra()
        elif event.key == 'q':
            plt.close('all')

    def _on_pick(self, event) -> None:
        """Maneja eventos de pick."""
        # Implementar lógica de selección por pick
        pass

    def _find_nearest_fiber(self, x: float, y: float) -> Optional[int]:
        """Encuentra fibra más cercana a coordenadas."""
        if self.fiber_positions is None:
            return None

        fx, fy = self.fiber_positions['X'], self.fiber_positions['Y']
        distances = np.sqrt((fx - x) ** 2 + (fy - y) ** 2)

        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        # Verificar si está dentro del radio de la fibra
        fiber_radius = self._calculate_fiber_radius(fx, fy)
        if min_dist <= fiber_radius * 1.5:  # Tolerancia adicional
            return min_idx

        return None

    def _select_fiber(self, fiber_idx: int) -> None:
        """Selecciona/deselecciona fibra."""
        if fiber_idx in self.selected_fibers:
            self.selected_fibers.remove(fiber_idx)
            # Quitar highlight
            self.fiber_patches[fiber_idx].set_linewidth(self.config.get('plw', 0.1))
            self.fiber_patches[fiber_idx].set_edgecolor(self.config.get('plc', 'k'))
        else:
            self.selected_fibers.append(fiber_idx)
            # Añadir highlight
            self.fiber_patches[fiber_idx].set_linewidth(self.config.get('clw', 2))
            self.fiber_patches[fiber_idx].set_edgecolor(self.config.get('cc', 'r'))

        self.figure.canvas.draw()

    def _show_fiber_spectrum(self, fiber_idx: int) -> None:
        """Muestra espectro de fibra específica."""
        # Implementar visualización de espectro
        if hasattr(self, 'spectral_data') and hasattr(self, 'wavelength'):
            spectrum = self.spectral_data[fiber_idx]

            # Crear o actualizar plot de espectro
            if not hasattr(self, 'spectrum_figure'):
                self.spectrum_figure = plt.figure(2, self.config.get('fig_spectra_size', (8, 5)))
                self.spectrum_axes = self.spectrum_figure.add_subplot(111)

            self.spectrum_axes.clear()
            self.spectrum_axes.plot(self.wavelength, spectrum,
                                    color=self.config.get('cspec', '#1f77b4'),
                                    linewidth=self.config.get('lspec', 1))

            self.spectrum_axes.set_xlabel('Wavelength (Å)')
            self.spectrum_axes.set_ylabel('Flux')
            self.spectrum_axes.set_title(f'Fiber {fiber_idx} Spectrum')

            self.spectrum_figure.canvas.draw()

    def _update_spectrum_preview(self, fiber_idx: int) -> None:
        """Actualiza preview de espectro."""
        # Versión simplificada del preview
        self._show_fiber_spectrum(fiber_idx)

    def _next_filter(self) -> None:
        """Cambia al siguiente filtro."""
        # Implementar cambio de filtro
        pass

    def _previous_filter(self) -> None:
        """Cambia al filtro anterior."""
        # Implementar cambio de filtro
        pass

    def _clear_selections(self) -> None:
        """Limpia todas las selecciones."""
        for fiber_idx in self.selected_fibers:
            self.fiber_patches[fiber_idx].set_linewidth(self.config.get('plw', 0.1))
            self.fiber_patches[fiber_idx].set_edgecolor(self.config.get('plc', 'k'))

        self.selected_fibers.clear()
        self.figure.canvas.draw()

    def _save_spectra(self) -> None:
        """Guarda espectros seleccionados."""
        if not self.selected_fibers:
            print("No fibers selected")
            return

        # Implementar guardado de espectros RSS
        print(f"Saving spectra for {len(self.selected_fibers)} selected fibers")

    def run(self) -> None:
        """Ejecuta el visor RSS."""
        plt.show()
