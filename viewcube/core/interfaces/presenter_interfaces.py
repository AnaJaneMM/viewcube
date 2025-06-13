"""
Interfaces para presentadores en ViewCube.

Este módulo define los contratos que deben implementar los presentadores
que formatean los datos para su visualización.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic, Tuple, Union
from PyQt5.QtWidgets import QWidget
import pyqtgraph as pg
import numpy as np

# Definición de tipos genéricos
T = TypeVar('T')
Result = TypeVar('Result')


class PresenterInterface(Generic[T, Result], ABC):
    """
    Interfaz base para todos los presentadores usando PyQt5.
    Define operaciones comunes que cualquier presentador debe implementar.
    """

    @abstractmethod
    def present(self, entity: T) -> Result:
        """
        Presenta una entidad utilizando widgets PyQt5.

        Args:
            entity: Entidad a presentar

        Returns:
            Widget PyQt5 o resultado formateado para visualización
        """
        pass

    @abstractmethod
    def present_error(self, error: Exception) -> Result:
        """
        Presenta un error usando elementos PyQt5.

        Args:
            error: Excepción a presentar

        Returns:
            Widget o mensaje formateado para mostrar el error
        """
        pass

    @abstractmethod
    def setup_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        """
        Configura el widget principal de presentación.

        Args:
            parent: Widget padre opcional

        Returns:
            Widget PyQt5 configurado
        """
        pass


class SpectrumPresenterInterface(ABC):
    """
    Interfaz para presentadores de datos espectrales usando PyQtGraph.
    Define operaciones para formatear y visualizar datos espectrales.
    """

    @abstractmethod
    def present_spectrum(self,
                         spectrum_data: Any,
                         plot_widget: pg.PlotWidget,
                         title: Optional[str] = None,
                         wavelength_range: Optional[Tuple[float, float]] = None,
                         flux_range: Optional[Tuple[float, float]] = None) -> pg.PlotDataItem:
        """
        Presenta datos de espectro en un PlotWidget de PyQtGraph.

        Args:
            spectrum_data: Datos del espectro
            plot_widget: Widget de gráfico PyQtGraph
            title: Título opcional
            wavelength_range: Rango de longitud de onda opcional
            flux_range: Rango de flujo opcional

        Returns:
            Elemento de datos ploteado
        """
        pass

    @abstractmethod
    def present_comparison(self,
                           spectrum1: Any,
                           spectrum2: Any,
                           plot_widget: pg.PlotWidget,
                           labels: Optional[Tuple[str, str]] = None) -> List[pg.PlotDataItem]:
        """
        Presenta una comparación de dos espectros en PyQtGraph.

        Args:
            spectrum1: Primer espectro
            spectrum2: Segundo espectro
            plot_widget: Widget de gráfico
            labels: Etiquetas para los espectros

        Returns:
            Lista de elementos ploteados
        """
        pass

    @abstractmethod
    def present_filter_response(self,
                                filter_data: Any,
                                plot_widget: pg.PlotWidget,
                                spectrum_data: Optional[Any] = None) -> pg.PlotDataItem:
        """
        Presenta la respuesta de un filtro usando PyQtGraph.

        Args:
            filter_data: Datos del filtro
            plot_widget: Widget de gráfico
            spectrum_data: Datos del espectro (opcional)

        Returns:
            Elemento de datos del filtro ploteado
        """
        pass

    @abstractmethod
    def format_spectrum_metadata(self, spectrum_data: Any) -> Dict[str, Any]:
        """
        Formatea los metadatos de un espectro para visualización en PyQt5.

        Args:
            spectrum_data: Datos del espectro

        Returns:
            Diccionario con metadatos formateados
        """
        pass

    @abstractmethod
    def configure_plot_style(self, plot_widget: pg.PlotWidget) -> None:
        """
        Configura el estilo del gráfico PyQtGraph.

        Args:
            plot_widget: Widget de gráfico a configurar
        """
        pass


class CubePresenterInterface(ABC):
    """
    Interfaz para presentadores de datos de cubo usando PyQtGraph.
    Define operaciones para formatear y visualizar datos de cubo espectral.
    """

    @abstractmethod
    def present_slice(self,
                      cube_data: Any,
                      image_widget: pg.ImageView,
                      wavelength_index: int,
                      colormap: Optional[str] = None,
                      scale: Optional[str] = None) -> pg.ImageItem:
        """
        Presenta un corte del cubo en PyQtGraph ImageView.

        Args:
            cube_data: Datos del cubo
            image_widget: Widget de imagen PyQtGraph
            wavelength_index: Índice de longitud de onda
            colormap: Mapa de colores opcional
            scale: Escala opcional (linear, log, etc.)

        Returns:
            Elemento de imagen
        """
        pass

    @abstractmethod
    def present_integrated_map(self,
                               cube_data: Any,
                               image_widget: pg.ImageView,
                               wavelength_range: Optional[Tuple[int, int]] = None,
                               colormap: Optional[str] = None,
                               scale: Optional[str] = None) -> pg.ImageItem:
        """
        Presenta un mapa integrado del cubo en PyQtGraph.

        Args:
            cube_data: Datos del cubo
            image_widget: Widget de imagen
            wavelength_range: Rango de índices de longitud de onda
            colormap: Mapa de colores opcional
            scale: Escala opcional

        Returns:
            Elemento de imagen integrada
        """
        pass

    @abstractmethod
    def present_spaxel_grid(self,
                            cube_data: Any,
                            layout_widget: QWidget,
                            positions: List[Tuple[int, int]],
                            wavelength_range: Optional[Tuple[float, float]] = None) -> List[pg.PlotWidget]:
        """
        Presenta una cuadrícula de espectros para múltiples spaxels.

        Args:
            cube_data: Datos del cubo
            layout_widget: Widget contenedor
            positions: Lista de posiciones (x, y)
            wavelength_range: Rango de longitud de onda opcional

        Returns:
            Lista de widgets de gráfico
        """
        pass

    @abstractmethod
    def format_cube_metadata(self, cube_data: Any) -> Dict[str, Any]:
        """
        Formatea los metadatos de un cubo para visualización.

        Args:
            cube_data: Datos del cubo

        Returns:
            Diccionario con metadatos formateados
        """
        pass

    @abstractmethod
    def setup_interactive_features(self,
                                   image_widget: pg.ImageView,
                                   plot_widget: pg.PlotWidget) -> None:
        """
        Configura características interactivas entre widgets.

        Args:
            image_widget: Widget de imagen
            plot_widget: Widget de gráfico
        """
        pass


class PyQtGraphPresenterMixin:
    """
    Mixin que proporciona funcionalidades comunes para presentadores PyQtGraph.
    """

    def create_plot_widget(self,
                           parent: Optional[QWidget] = None,
                           background: str = 'w',
                           title: Optional[str] = None) -> pg.PlotWidget:
        """
        Crea un widget de gráfico PyQtGraph configurado.

        Args:
            parent: Widget padre
            background: Color de fondo
            title: Título del gráfico

        Returns:
            Widget de gráfico configurado
        """
        plot_widget = pg.PlotWidget(parent=parent, background=background)

        if title:
            plot_widget.setTitle(title)

        plot_widget.setLabel('left', 'Flux')
        plot_widget.setLabel('bottom', 'Wavelength (Å)')
        plot_widget.showGrid(x=True, y=True, alpha=0.3)

        return plot_widget

    def create_image_widget(self,
                            parent: Optional[QWidget] = None) -> pg.ImageView:
        """
        Crea un widget de imagen PyQtGraph configurado.

        Args:
            parent: Widget padre

        Returns:
            Widget de imagen configurado
        """
        image_widget = pg.ImageView(parent=parent)
        image_widget.ui.roiBtn.hide()
        image_widget.ui.menuBtn.hide()

        return image_widget

    def apply_colormap(self,
                       image_item: pg.ImageItem,
                       colormap: str = 'viridis') -> None:
        """
        Aplica un mapa de colores a un elemento de imagen.

        Args:
            image_item: Elemento de imagen
            colormap: Nombre del mapa de colores
        """
        try:
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors

            # Crear mapa de colores personalizado para PyQtGraph
            cmap = cm.get_cmap(colormap)
            colors = []
            for i in range(256):
                rgba = cmap(i / 255.0)
                colors.append([int(rgba[0] * 255), int(rgba[1] * 255),
                               int(rgba[2] * 255), int(rgba[3] * 255)])

            # Aplicar mapa de colores
            image_item.setLookupTable(colors)

        except ImportError:
            # Fallback a escalas de grises si matplotlib no está disponible
            pass


class InteractivePresenterInterface(ABC):
    """
    Interfaz para presentadores con capacidades interactivas.
    """

    @abstractmethod
    def setup_mouse_interaction(self, widget: QWidget) -> None:
        """
        Configura interacciones con el mouse.

        Args:
            widget: Widget donde configurar las interacciones
        """
        pass

    @abstractmethod
    def setup_keyboard_shortcuts(self, widget: QWidget) -> None:
        """
        Configura atajos de teclado.

        Args:
            widget: Widget donde configurar los atajos
        """
        pass

    @abstractmethod
    def handle_selection_change(self, selection_data: Any) -> None:
        """
        Maneja cambios en la selección.

        Args:
            selection_data: Datos de la selección
        """
        pass