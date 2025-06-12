"""
Interfaces para servicios en ViewCube.

Este módulo define los contratos que deben implementar los servicios
que contienen la lógica de negocio de la aplicación.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic, Tuple

# Definición de tipos genéricos
T = TypeVar('T')  # Tipo de entidad
DTO = TypeVar('DTO')  # Tipo de objeto de transferencia de datos


class ServiceInterface(Generic[T, DTO], ABC):
    """
    Interfaz base para todos los servicios.

    Define operaciones comunes que cualquier servicio debe implementar.
    """

    @abstractmethod
    def get_by_id(self, id: Any) -> Optional[DTO]:
        """
        Obtiene una entidad por su identificador.

        Args:
            id: Identificador único de la entidad

        Returns:
            DTO de la entidad o None si no existe
        """
        pass

    @abstractmethod
    def get_all(self) -> List[DTO]:
        """
        Obtiene todas las entidades.

        Returns:
            Lista de DTOs de todas las entidades
        """
        pass


class DataServiceInterface(ABC):
    """
    Interfaz para servicios que manejan datos astronómicos.

    Define operaciones para cargar, procesar y manipular
    datos de cubos espectrales y espectros.
    """

    @abstractmethod
    def load_fits_file(self, filename: str, **kwargs) -> Dict[str, Any]:
        """
        Carga un archivo FITS y extrae toda la información relevante.

        Args:
            filename: Ruta al archivo FITS
            **kwargs: Parámetros adicionales (extensiones, etc.)

        Returns:
            Diccionario con los datos extraídos del archivo
        """
        pass

    @abstractmethod
    def create_spectrum_data(self, wavelength: List[float], flux: List[float],
                             error: Optional[List[float]] = None,
                             flag: Optional[List[bool]] = None,
                             metadata: Optional[Dict] = None) -> Any:
        """
        Crea un objeto de datos de espectro.

        Args:
            wavelength: Array de longitudes de onda
            flux: Array de flujo
            error: Array de errores (opcional)
            flag: Array de flags (opcional)
            metadata: Metadatos adicionales

        Returns:
            Objeto SpectrumData
        """
        pass

    @abstractmethod
    def create_cube_data(self, data: Any,
                         wavelength: Optional[List[float]] = None,
                         error: Optional[Any] = None,
                         flag: Optional[Any] = None,
                         metadata: Optional[Dict] = None) -> Any:
        """
        Crea un objeto de datos de cubo.

        Args:
            data: Array 3D de datos (lambda, y, x)
            wavelength: Array de longitudes de onda
            error: Array de errores
            flag: Array de flags
            metadata: Metadatos adicionales

        Returns:
            Objeto CubeData
        """
        pass

    @abstractmethod
    def extract_spectrum_from_cube(self, cube_data: Any, x: int, y: int) -> Optional[Any]:
        """
        Extrae un espectro de un spaxel específico del cubo.

        Args:
            cube_data: Objeto CubeData
            x: Coordenada X del spaxel
            y: Coordenada Y del spaxel

        Returns:
            Objeto SpectrumData del spaxel o None si las coordenadas son inválidas
        """
        pass

    @abstractmethod
    def calculate_integrated_spectrum(self, cube_data: Any,
                                      mask: Optional[Any] = None) -> Any:
        """
        Calcula el espectro integrado de un cubo.

        Args:
            cube_data: Objeto CubeData
            mask: Máscara opcional para seleccionar spaxels

        Returns:
            Espectro integrado como SpectrumData
        """
        pass

    @abstractmethod
    def apply_velocity_correction(self, wavelength: List[float], velocity: float) -> List[float]:
        """
        Aplica corrección de velocidad a longitudes de onda.

        Args:
            wavelength: Array de longitudes de onda
            velocity: Velocidad en km/s

        Returns:
            Array de longitudes de onda corregidas
        """
        pass


class FilterServiceInterface(ABC):
    """
    Interfaz para servicios que manejan filtros espectrales.

    Define operaciones para cargar, aplicar y procesar filtros.
    """

    @abstractmethod
    def list_available_filters(self) -> List[str]:
        """
        Lista todos los filtros disponibles.

        Returns:
            Lista de nombres de filtros disponibles
        """
        pass

    @abstractmethod
    def load_filter(self, filter_name: str, force_reload: bool = False) -> Any:
        """
        Carga un filtro desde archivo.

        Args:
            filter_name: Nombre del filtro
            force_reload: Forzar recarga aunque esté en caché

        Returns:
            Objeto FilterData
        """
        pass

    @abstractmethod
    def apply_filter_to_spectrum(self, spectrum: Any,
                                 filter_data: Any,
                                 normalize: bool = True) -> Any:
        """
        Aplica un filtro a un espectro.

        Args:
            spectrum: Espectro de entrada
            filter_data: Datos del filtro
            normalize: Normalizar por la respuesta del filtro

        Returns:
            Espectro filtrado
        """
        pass

    @abstractmethod
    def apply_filter_to_cube(self, cube: Any,
                             filter_data: Any,
                             normalize: bool = True) -> Any:
        """
        Aplica un filtro a un cubo espectral completo.

        Args:
            cube: Cubo de datos
            filter_data: Datos del filtro
            normalize: Normalizar por la respuesta del filtro

        Returns:
            Imagen filtrada (2D array)
        """
        pass

    @abstractmethod
    def calculate_passband_limits(self, filter_data: Any,
                                  threshold: float = 0.1) -> Tuple[float, float]:
        """
        Calcula los límites efectivos de la banda de paso.

        Args:
            filter_data: Datos del filtro
            threshold: Umbral de respuesta (fracción del máximo)

        Returns:
            Tupla (wavelength_min, wavelength_max)
        """
        pass


class SonificationServiceInterface(ABC):
    """
    Interfaz para servicios que manejan sonificación.

    Define operaciones para convertir datos astronómicos en audio.
    """

    @abstractmethod
    def initialize_sonification(self) -> bool:
        """
        Inicializa el sistema de sonificación.

        Returns:
            True si la inicialización fue exitosa
        """
        pass

    @abstractmethod
    def setup_cube_sonification(self, cube_data: Any,
                                fits_file: str,
                                reference_position: Optional[Tuple[int, int]] = None,
                                matplotlib_figure=None) -> bool:
        """
        Configura la sonificación para un cubo de datos.

        Args:
            cube_data: Datos del cubo
            fits_file: Archivo FITS original
            reference_position: Posición de referencia (y, x)
            matplotlib_figure: Figura de matplotlib para eventos

        Returns:
            True si la configuración fue exitosa
        """
        pass

    @abstractmethod
    def sonify_spaxel(self, x: int, y: int, spectrum: Optional[Any] = None) -> bool:
        """
        Sonifica un spaxel específico.

        Args:
            x: Coordenada X del spaxel
            y: Coordenada Y del spaxel
            spectrum: Datos del espectro (opcional)

        Returns:
            True si la sonificación fue exitosa
        """
        pass

    @abstractmethod
    def stop_sonification(self) -> None:
        """Detiene toda la sonificación activa."""
        pass


class EventServiceInterface(ABC):
    """
    Interfaz para servicios que manejan eventos.

    Define operaciones para registrar, emitir y gestionar eventos
    del sistema y la interfaz de usuario.
    """

    @abstractmethod
    def register_handler(self, event_type: str, handler: callable) -> None:
        """
        Registra un manejador para un tipo de evento.

        Args:
            event_type: Tipo de evento
            handler: Función a ejecutar cuando ocurra el evento
        """
        pass

    @abstractmethod
    def unregister_handler(self, event_type: str, handler: callable) -> None:
        """
        Desregistra un manejador de eventos.

        Args:
            event_type: Tipo de evento
            handler: Función a desregistrar
        """
        pass

    @abstractmethod
    def emit_event(self, event_type: str, data: Any) -> None:
        """
        Emite un evento con datos.

        Args:
            event_type: Tipo de evento
            data: Datos del evento
        """
        pass

    @abstractmethod
    def setup_matplotlib_events(self, figure, figure_name: str) -> None:
        """
        Configura los eventos de matplotlib para una figura.

        Args:
            figure: Figura de matplotlib
            figure_name: Nombre identificador de la figura
        """
        pass
