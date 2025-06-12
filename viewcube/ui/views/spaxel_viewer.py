"""Visualizador de spaxels con gestión de eventos."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from typing import Tuple, List, Optional, Callable
from abc import ABC, abstractmethod


class ViewerInterface(ABC):
    """Interfaz base para visualizadores."""

    @abstractmethod
    def setup_figure(self) -> None:
        pass

    @abstractmethod
    def update_display(self) -> None:
        pass

    @abstractmethod
    def clear_display(self) -> None:
        pass


class SpaxelViewer(ViewerInterface):
    """Visualizador especializado para spaxels."""

    def __init__(self, figure_size: Tuple[float, float] = (7.1, 6)):
        self.figure_size = figure_size
        self.figure = None
        self.axes = None
        self.image_plot = None
        self.colorbar = None

        # Propiedades de visualización
        self.alpha = 0.95
        self.extent = None
        self.color_data = None

        # Lista de elementos seleccionados
        self.selected_elements: List[int] = []
        self.selection_patches: List = []

        # Callbacks para eventos
        self.on_click_callback: Optional[Callable] = None
        self.on_motion_callback: Optional[Callable] = None

        self.setup_figure()

    def setup_figure(self) -> None:
        """Configura la figura y ejes principales."""
        self.figure = plt.figure(1, self.figure_size)
        self.figure.set_label("Spaxel Viewer")
        self.figure.canvas.manager.set_window_title("Spaxel Viewer")
        self.axes = self.figure.add_subplot(111)

        # Conectar eventos
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def set_image_data(self, data: np.ndarray, extent: List[float]) -> None:
        """Establece datos de imagen para visualización."""
        self.color_data = data
        self.extent = extent
        self.update_display()

    def update_display(self) -> None:
        """Actualiza la visualización principal."""
        if self.color_data is None:
            return

        # Limpiar plot anterior si existe
        if self.image_plot:
            self.image_plot.remove()

        # Crear nueva visualización
        from .rgbmpl import rnorm
        self.image_plot = self.axes.imshow(
            self.color_data,
            alpha=self.alpha,
            extent=self.extent,
            norm=rnorm("sqrt"),
            interpolation="nearest",
            aspect="auto",
            origin="lower"
        )

        self.axes.axis(self.extent)
        self._update_color_limits()
        self.figure.canvas.draw()

    def _update_color_limits(self) -> None:
        """Actualiza límites de color automáticamente."""
        if self.color_data is not None:
            vmin, vmax = np.nanmin(self.color_data), np.nanmax(self.color_data)
            if np.isfinite(vmin) and np.isfinite(vmax):
                self.image_plot.set_clim([vmin, vmax])

    def add_colorbar(self) -> None:
        """Añade barra de colores."""
        if self.image_plot and not self.colorbar:
            self.colorbar = self.figure.colorbar(self.image_plot)

    def clear_display(self) -> None:
        """Limpia la visualización."""
        self.axes.clear()
        self.selected_elements.clear()
        self.selection_patches.clear()
        self.image_plot = None
        self.figure.canvas.draw()

    def add_selection(self, element_id: int, position: Tuple[float, float],
                      radius: float, color: str = "red") -> None:
        """Añade elemento seleccionado."""
        if element_id not in self.selected_elements:
            self.selected_elements.append(element_id)

            # Crear patch visual
            circle = Circle(position, radius, color=color, fill=False, linewidth=2)
            self.axes.add_patch(circle)
            self.selection_patches.append(circle)

            self.figure.canvas.draw()

    def remove_selection(self, element_id: int) -> None:
        """Remueve elemento seleccionado."""
        if element_id in self.selected_elements:
            index = self.selected_elements.index(element_id)
            self.selected_elements.remove(element_id)

            # Remover patch visual
            if index < len(self.selection_patches):
                self.selection_patches[index].remove()
                self.selection_patches.pop(index)

            self.figure.canvas.draw()

    def clear_selections(self) -> None:
        """Limpia todas las selecciones."""
        self.selected_elements.clear()
        for patch in self.selection_patches:
            patch.remove()
        self.selection_patches.clear()
        self.figure.canvas.draw()

    def set_click_callback(self, callback: Callable) -> None:
        """Establece callback para eventos de click."""
        self.on_click_callback = callback

    def set_motion_callback(self, callback: Callable) -> None:
        """Establece callback para eventos de movimiento."""
        self.on_motion_callback = callback

    def _on_click(self, event) -> None:
        """Maneja eventos de click."""
        if self.on_click_callback and event.inaxes == self.axes:
            self.on_click_callback(event)

    def _on_motion(self, event) -> None:
        """Maneja eventos de movimiento del mouse."""
        if self.on_motion_callback and event.inaxes == self.axes:
            self.on_motion_callback(event)

    def set_title(self, title: str) -> None:
        """Establece título del viewer."""
        self.axes.set_title(title)
        self.figure.canvas.draw()
