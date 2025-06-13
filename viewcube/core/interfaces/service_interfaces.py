"""
Interfaces para servicios en ViewCube.

Este módulo define los contratos que deben implementar los servicios
que contienen la lógica de negocio de la aplicación.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic, Tuple, Union, Callable
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

# Definición de tipos genéricos
T = TypeVar('T')
DTO = TypeVar('DTO')


class ServiceInterface(Generic[T, DTO], ABC):
    """
    Interfaz base para todos los servicios.
    Define operaciones comunes independientes de la capa de presentación.
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
    def get_all(self, **criteria) -> List[DTO]:
        """
        Obtiene todas las entidades que cumplen los criterios.

        Args:
            **criteria: Criterios de filtrado

        Returns:
            Lista de DTOs de las entidades
        """
        pass

    @abstractmethod
    def create(self, data: Dict[str, Any]) -> DTO:
        """
        Crea una nueva entidad.

        Args:
            data: Datos para crear la entidad

        Returns:
            DTO de la entidad creada
        """
        pass

    @abstractmethod
    def update(self, id: Any, data: Dict[str, Any]) -> Optional[DTO]:
        """
        Actualiza una entidad existente.

        Args:
            id: Identificador de la entidad
            data: Datos de actualización

        Returns:
            DTO de la entidad actualizada o None si no existe
        """
        pass

    @abstractmethod
    def delete(self, id: Any) -> bool:
        """
        Elimina una entidad.

        Args:
            id: Identificador de la entidad

        Returns:
            True si se eliminó correctamente
        """
        pass


class DataServiceInterface(ABC):
    """
    Interfaz para servicios que manejan datos astronómicos.
    Lógica de negocio pura sin dependencias de presentación.
    """

    @abstractmethod
    def load_fits_file(self,
                       filename: Union[str, Path],
                       **kwargs) -> Dict[str, Any]:
        """
        Carga un archivo FITS y procesa la información.

        Args:
            filename: Ruta al archivo FITS
            **kwargs: Parámetros adicionales

        Returns:
            Diccionario con los datos procesados
        """
        pass

    @abstractmethod
    def create_spectrum_data(self,
                             wavelength: np.ndarray,
                             flux: np.ndarray,
                             error: Optional[np.ndarray] = None,
                             flag: Optional[np.ndarray] = None,
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
    def create_cube_data(self,
                         data: np.ndarray,
                         wavelength: Optional[np.ndarray] = None,
                         error: Optional[np.ndarray] = None,
                         flag: Optional[np.ndarray] = None,
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
    def extract_spectrum_from_cube(self,
                                   cube_data: Any,
                                   x: int,
                                   y: int) -> Optional[Any]:
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
    def calculate_integrated_spectrum(self,
                                      cube_data: Any,
                                      mask: Optional[np.ndarray] = None) -> Any:
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
    def apply_velocity_correction(self,
                                  wavelength: np.ndarray,
                                  velocity: float) -> np.ndarray:
        """
        Aplica corrección de velocidad a longitudes de onda.

        Args:
            wavelength: Array de longitudes de onda
            velocity: Velocidad en km/s

        Returns:
            Array de longitudes de onda corregidas
        """
        pass

    @abstractmethod
    def validate_data_integrity(self, data: Any) -> Tuple[bool, List[str]]:
        """
        Valida la integridad de los datos.

        Args:
            data: Datos a validar

        Returns:
            Tupla (es_válido, lista_de_errores)
        """
        pass

    @abstractmethod
    def calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calcula estadísticas básicas de los datos.

        Args:
            data: Array de datos

        Returns:
            Diccionario con estadísticas calculadas
        """
        pass


class FilterServiceInterface(ABC):
    """
    Interfaz para servicios que manejan filtros espectrales.
    Lógica de negocio para procesamiento de filtros.
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
    def load_filter(self,
                    filter_name: str,
                    force_reload: bool = False) -> Any:
        """
        Carga un filtro desde repositorio.

        Args:
            filter_name: Nombre del filtro
            force_reload: Forzar recarga aunque esté en caché

        Returns:
            Objeto FilterData
        """
        pass

    @abstractmethod
    def apply_filter_to_spectrum(self,
                                 spectrum: Any,
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
    def apply_filter_to_cube(self,
                             cube: Any,
                             filter_data: Any,
                             normalize: bool = True) -> np.ndarray:
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
    def calculate_passband_limits(self,
                                  filter_data: Any,
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

    @abstractmethod
    def convolve_spectrum(self,
                          spectrum: Any,
                          filter_data: Any) -> Any:
        """
        Convoluciona un espectro con un filtro.

        Args:
            spectrum: Espectro de entrada
            filter_data: Datos del filtro

        Returns:
            Espectro convolucionado
        """
        pass

    @abstractmethod
    def interpolate_filter(self,
                           filter_data: Any,
                           new_wavelength: np.ndarray) -> Any:
        """
        Interpola un filtro a una nueva grilla de longitudes de onda.

        Args:
            filter_data: Datos del filtro original
            new_wavelength: Nueva grilla de longitudes de onda

        Returns:
            Filtro interpolado
        """
        pass


class SonificationServiceInterface(QObject):
    """
    Interfaz para servicios de sonificación usando señales PyQt5.
    Maneja conversión de datos astronómicos a audio.
    """

    # Señales PyQt5 para comunicación asíncrona
    sonification_started = pyqtSignal()
    sonification_stopped = pyqtSignal()
    sonification_error = pyqtSignal(str)
    position_changed = pyqtSignal(int, int)

    @abstractmethod
    def initialize_sonification(self) -> bool:
        """
        Inicializa el sistema de sonificación.

        Returns:
            True si la inicialización fue exitosa
        """
        pass

    @abstractmethod
    def setup_cube_sonification(self,
                                cube_data: Any,
                                fits_file: Union[str, Path],
                                reference_position: Optional[Tuple[int, int]] = None) -> bool:
        """
        Configura la sonificación para un cubo de datos.

        Args:
            cube_data: Datos del cubo
            fits_file: Archivo FITS original
            reference_position: Posición de referencia (y, x)

        Returns:
            True si la configuración fue exitosa
        """
        pass

    @abstractmethod
    def sonify_spaxel(self,
                      x: int,
                      y: int,
                      spectrum: Optional[Any] = None) -> bool:
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

    @abstractmethod
    def set_sonification_parameters(self, params: Dict[str, Any]) -> None:
        """
        Configura parámetros de sonificación.

        Args:
            params: Diccionario con parámetros
        """
        pass

    @abstractmethod
    def get_sonification_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual de la sonificación.

        Returns:
            Diccionario con el estado
        """
        pass


class EventServiceInterface(QObject):
    """
    Interfaz para servicios que manejan eventos usando PyQt5.
    Define operaciones para gestión de eventos del sistema.
    """

    # Señales PyQt5 para diferentes tipos de eventos
    data_loaded = pyqtSignal(dict)
    spectrum_selected = pyqtSignal(int, int)
    filter_changed = pyqtSignal(str)
    view_updated = pyqtSignal()
    error_occurred = pyqtSignal(str)

    @abstractmethod
    def register_handler(self,
                         event_type: str,
                         handler: Callable[[Any], None]) -> None:
        """
        Registra un manejador para un tipo de evento.

        Args:
            event_type: Tipo de evento
            handler: Función a ejecutar cuando ocurra el evento
        """
        pass

    @abstractmethod
    def unregister_handler(self,
                           event_type: str,
                           handler: Callable[[Any], None]) -> None:
        """
        Desregistra un manejador de eventos.

        Args:
            event_type: Tipo de evento
            handler: Función a desregistrar
        """
        pass

    @abstractmethod
    def emit_event(self,
                   event_type: str,
                   data: Any) -> None:
        """
        Emite un evento con datos.

        Args:
            event_type: Tipo de evento
            data: Datos del evento
        """
        pass

    @abstractmethod
    def setup_qt_connections(self,
                             source_widget: QObject,
                             target_handler: Callable) -> None:
        """
        Configura conexiones de señales PyQt5.

        Args:
            source_widget: Widget fuente de la señal
            target_handler: Manejador de la señal
        """
        pass

    @abstractmethod
    def disconnect_all(self) -> None:
        """Desconecta todos los manejadores de eventos."""
        pass


class ValidationServiceInterface(ABC):
    """
    Interfaz para servicios de validación.
    """

    @abstractmethod
    def validate_fits_file(self,
                           filename: Union[str, Path]) -> Tuple[bool, List[str]]:
        """
        Valida un archivo FITS.

        Args:
            filename: Ruta al archivo

        Returns:
            Tupla (es_válido, lista_de_errores)
        """
        pass

    @abstractmethod
    def validate_spectrum_data(self,
                               wavelength: np.ndarray,
                               flux: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Valida datos de espectro.

        Args:
            wavelength: Array de longitudes de onda
            flux: Array de flujo

        Returns:
            Tupla (es_válido, lista_de_errores)
        """
        pass

    @abstractmethod
    def validate_cube_data(self, data: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Valida datos de cubo.

        Args:
            data: Array 3D de datos

        Returns:
            Tupla (es_válido, lista_de_errores)
        """
        pass


class CacheServiceInterface(ABC):
    """
    Interfaz para servicios de caché.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del caché.

        Args:
            key: Clave del valor

        Returns:
            Valor almacenado o None
        """
        pass

    @abstractmethod
    def put(self,
            key: str,
            value: Any,
            ttl: Optional[int] = None) -> None:
        """
        Almacena un valor en el caché.

        Args:
            key: Clave del valor
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos
        """
        pass

    @abstractmethod
    def invalidate(self, pattern: Optional[str] = None) -> None:
        """
        Invalida entradas del caché.

        Args:
            pattern: Patrón de claves a invalidar (None para todo)
        """
        pass


class CalculationServiceInterface(ABC):
    """
    Interfaz para servicios de cálculos científicos.
    """

    @abstractmethod
    def calculate_continuum(self,
                            spectrum: Any,
                            method: str = 'polynomial',
                            degree: int = 3) -> np.ndarray:
        """
        Calcula el continuo de un espectro.

        Args:
            spectrum: Datos del espectro
            method: Método de cálculo
            degree: Grado del polinomio (si aplica)

        Returns:
            Array con el continuo calculado
        """
        pass

    @abstractmethod
    def measure_equivalent_width(self,
                                 spectrum: Any,
                                 line_center: float,
                                 continuum_regions: List[Tuple[float, float]]) -> float:
        """
        Mide el ancho equivalente de una línea espectral.

        Args:
            spectrum: Datos del espectro
            line_center: Centro de la línea
            continuum_regions: Regiones para definir el continuo

        Returns:
            Ancho equivalente medido
        """
        pass

    @abstractmethod
    def fit_gaussian(self,
                     spectrum: Any,
                     center_guess: float,
                     width_guess: float) -> Dict[str, float]:
        """
        Ajusta una gaussiana a una línea espectral.

        Args:
            spectrum: Datos del espectro
            center_guess: Estimación inicial del centro
            width_guess: Estimación inicial del ancho

        Returns:
            Diccionario con parámetros del ajuste
        """
        pass