"""
Interfaces para presentadores en ViewCube.

Este módulo define los contratos que deben implementar los presentadores
que formatean los datos para su visualización.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic, Tuple

# Definición de tipos genéricos
T = TypeVar('T')  # Tipo de entidad
Result = TypeVar('Result')  # Tipo de resultado


class PresenterInterface(Generic[T, Result], ABC):
    """
    Interfaz base para todos los presentadores.

    Define operaciones comunes que cualquier presentador debe implementar.
    """

    @abstractmethod
    def present(self, entity: T) -> Result:
        """
        Presenta una entidad.

        Args:
            entity: Entidad a presentar

        Returns:
            Resultado formateado para visualización
        """
        pass

    @abstractmethod
    def present_error(self, error: Exception) -> Result:
        """
        Presenta un error.

        Args:
            error: Excepción a presentar

        Returns:
            Resultado formateado para visualización
        """
        pass


class SpectrumPresenterInterface(ABC):
    """
    Interfaz para presentadores que manejan datos de espectro.

    Define operaciones para formatear y visualizar datos espectrales.
    """

    @abstractmethod
    def present_spectrum(self, spectrum_data: Any,
                         title: Optional[str] = None,
                         wavelength_range: Optional[Tuple[float, float]] = None,
                         flux_range: Optional[Tuple[float, float]] = None) -> Any:
        """
        Presenta datos de espectro.

        Args:
            spectrum_data: Datos del espectro
            title: Título opcional
            wavelength_range: Rango de longitud de onda opcional
            flux_range: Rango de flujo opcional

        Returns:
            Objeto de visualización (figura, datos formateados, etc.)
        """
        pass

    @abstractmethod
    def present_comparison(self, spectrum1: Any, spectrum2: Any,
                           labels: Optional[Tuple[str, str]] = None) -> Any:
        """
        Presenta una comparación de dos espectros.

        Args:
            spectrum1: Primer espectro
            spectrum2: Segundo espectro
            labels: Etiquetas para los espectros

        Returns:
            Objeto de visualización
        """
        pass

    @abstractmethod
    def present_filter_response(self, filter_data: Any,
                                spectrum_data: Optional[Any] = None) -> Any:
        """
        Presenta la respuesta de un filtro, opcionalmente superpuesta a un espectro.

        Args:
            filter_data: Datos del filtro
            spectrum_data: Datos del espectro (opcional)

        Returns:
            Objeto de visualización
        """
        pass

    @abstractmethod
    def format_spectrum_metadata(self, spectrum_data: Any) -> Dict[str, Any]:
        """
        Formatea los metadatos de un espectro para visualización.

        Args:
            spectrum_data: Datos del espectro

        Returns:
            Diccionario con metadatos formateados
        """
        pass


class CubePresenterInterface(ABC):
    """
    Interfaz para presentadores que manejan datos de cubo.

    Define operaciones para formatear y visualizar datos de cubo espectral.
    """

    @abstractmethod
    def present_slice(self, cube_data: Any,
                      wavelength_index: int,
                      colormap: Optional[str] = None,
                      scale: Optional[str] = None) -> Any:
        """
        Presenta un corte del cubo a una longitud de onda específica.

        Args:
            cube_data: Datos del cubo
            wavelength_index: Índice de longitud de onda
            colormap: Mapa de colores opcional
            scale: Escala opcional (linear, log, etc.)

        Returns:
            Objeto de visualización
        """
        pass

    @abstractmethod
    def present_integrated_map(self, cube_data: Any,
                               wavelength_range: Optional[Tuple[int, int]] = None,
                               colormap: Optional[str] = None,
                               scale: Optional[str] = None) -> Any:
        """
        Presenta un mapa integrado del cubo en un rango de longitudes de onda.

        Args:
            cube_data: Datos del cubo
            wavelength_range: Rango de índices de longitud de onda
            colormap: Mapa de colores opcional
            scale: Escala opcional (linear, log, etc.)

        Returns:
            Objeto de visualización
        """
        pass

    @abstractmethod
    def present_spaxel_grid(self, cube_data: Any,
                            positions: List[Tuple[int, int]],
                            wavelength_range: Optional[Tuple[float, float]] = None) -> Any:
        """
        Presenta una cuadrícula de espectros para múltiples spaxels.

        Args:
            cube_data: Datos del cubo
            positions: Lista de posiciones (x, y)
            wavelength_range: Rango de longitud de onda opcional

        Returns:
            Objeto de visualización
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
