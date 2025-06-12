"""
Interfaces para repositorios en ViewCube.

Este módulo define los contratos que deben implementar los repositorios
para acceder a diferentes fuentes de datos.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic

# Definición de tipos genéricos
T = TypeVar('T')  # Tipo de entidad
ID = TypeVar('ID')  # Tipo de identificador


class RepositoryInterface(Generic[T, ID], ABC):
    """
    Interfaz base para todos los repositorios.

    Define las operaciones básicas que cualquier repositorio debe implementar,
    independientemente del tipo de almacenamiento.
    """

    @abstractmethod
    def find_by_id(self, id: ID) -> Optional[T]:
        """
        Busca una entidad por su identificador.

        Args:
            id: Identificador único de la entidad

        Returns:
            La entidad encontrada o None si no existe
        """
        pass

    @abstractmethod
    def find_all(self) -> List[T]:
        """
        Recupera todas las entidades.

        Returns:
            Lista de todas las entidades
        """
        pass

    @abstractmethod
    def save(self, entity: T) -> T:
        """
        Guarda una entidad (crea o actualiza).

        Args:
            entity: Entidad a guardar

        Returns:
            La entidad guardada
        """
        pass

    @abstractmethod
    def delete(self, id: ID) -> None:
        """
        Elimina una entidad por su identificador.

        Args:
            id: Identificador único de la entidad
        """
        pass


class FitsRepositoryInterface(ABC):
    """
    Interfaz para repositorios que manejan archivos FITS.

    Define operaciones específicas para la carga, procesamiento y
    almacenamiento de datos astronómicos en formato FITS.
    """

    @abstractmethod
    def load_fits_file(self, filename: str, **kwargs) -> Dict[str, Any]:
        """
        Carga un archivo FITS y extrae su información.

        Args:
            filename: Ruta al archivo FITS
            **kwargs: Parámetros adicionales (extensiones, etc.)

        Returns:
            Diccionario con los datos extraídos del archivo
        """
        pass

    @abstractmethod
    def extract_wavelength(self, header: Dict[str, Any], specaxis: int = None) -> Optional[List[float]]:
        """
        Extrae información de longitud de onda del header FITS.

        Args:
            header: Header del archivo FITS
            specaxis: Eje espectral (opcional)

        Returns:
            Array de longitudes de onda o None si no se puede extraer
        """
        pass

    @abstractmethod
    def extract_metadata(self, header: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae metadatos del header FITS.

        Args:
            header: Header del archivo FITS

        Returns:
            Diccionario con metadatos extraídos
        """
        pass

    @abstractmethod
    def save_spectrum(self, wavelength: List[float], flux: List[float],
                      filename: str, header: Optional[Dict[str, Any]] = None) -> None:
        """
        Guarda un espectro en formato FITS.

        Args:
            wavelength: Array de longitudes de onda
            flux: Array de flujo
            filename: Nombre del archivo de salida
            header: Header FITS opcional
        """
        pass


class ConfigRepositoryInterface(ABC):
    """
    Interfaz para repositorios que manejan configuración.

    Define operaciones para cargar, guardar y gestionar
    configuraciones de la aplicación.
    """

    @abstractmethod
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Carga configuración desde un archivo.

        Args:
            config_path: Ruta al archivo de configuración (opcional)

        Returns:
            Diccionario con la configuración cargada
        """
        pass

    @abstractmethod
    def save_config(self, config: Dict[str, Any], config_path: Optional[str] = None) -> None:
        """
        Guarda configuración en un archivo.

        Args:
            config: Configuración a guardar
            config_path: Ruta al archivo de configuración (opcional)
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración por defecto.

        Returns:
            Diccionario con la configuración por defecto
        """
        pass

    @abstractmethod
    def merge_config(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combina configuración por defecto con configuración de usuario.

        Args:
            default_config: Configuración por defecto
            user_config: Configuración de usuario

        Returns:
            Configuración combinada
        """
        pass
