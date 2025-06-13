"""
Interfaces para repositorios en ViewCube.

Este módulo define los contratos que deben implementar los repositorios
para acceder a diferentes fuentes de datos.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic, Union, Tuple
from pathlib import Path
import numpy as np

# Definición de tipos genéricos
T = TypeVar('T')
ID = TypeVar('ID')


class RepositoryInterface(Generic[T, ID], ABC):
    """
    Interfaz base para todos los repositorios.
    Define operaciones básicas CRUD independientes del almacenamiento.
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
    def find_all(self, **criteria) -> List[T]:
        """
        Recupera entidades que cumplan los criterios especificados.

        Args:
            **criteria: Criterios de búsqueda

        Returns:
            Lista de entidades que cumplen los criterios
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
    def delete(self, id: ID) -> bool:
        """
        Elimina una entidad por su identificador.

        Args:
            id: Identificador único de la entidad

        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        pass

    @abstractmethod
    def exists(self, id: ID) -> bool:
        """
        Verifica si existe una entidad con el identificador dado.

        Args:
            id: Identificador a verificar

        Returns:
            True si existe, False en caso contrario
        """
        pass

    @abstractmethod
    def count(self, **criteria) -> int:
        """
        Cuenta entidades que cumplen los criterios.

        Args:
            **criteria: Criterios de búsqueda

        Returns:
            Número de entidades que cumplen los criterios
        """
        pass


class FitsRepositoryInterface(ABC):
    """
    Interfaz para repositorios que manejan archivos FITS.
    Define operaciones específicas para datos astronómicos.
    """

    @abstractmethod
    def load_fits_file(self,
                       filename: Union[str, Path],
                       **kwargs) -> Dict[str, Any]:
        """
        Carga un archivo FITS y extrae su información.

        Args:
            filename: Ruta al archivo FITS
            **kwargs: Parámetros adicionales (extensiones, etc.)

        Returns:
            Diccionario con los datos extraídos del archivo

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el archivo no es un FITS válido
        """
        pass

    @abstractmethod
    def extract_wavelength_solution(self,
                                    header: Dict[str, Any],
                                    specaxis: Optional[int] = None) -> Optional[np.ndarray]:
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
            Diccionario con metadatos extraídos y procesados
        """
        pass

    @abstractmethod
    def save_spectrum(self,
                      wavelength: np.ndarray,
                      flux: np.ndarray,
                      filename: Union[str, Path],
                      header: Optional[Dict[str, Any]] = None,
                      error: Optional[np.ndarray] = None) -> None:
        """
        Guarda un espectro en formato FITS.

        Args:
            wavelength: Array de longitudes de onda
            flux: Array de flujo
            filename: Nombre del archivo de salida
            header: Header FITS opcional
            error: Array de errores opcional

        Raises:
            IOError: Si no se puede escribir el archivo
        """
        pass

    @abstractmethod
    def save_cube(self,
                  data: np.ndarray,
                  filename: Union[str, Path],
                  header: Optional[Dict[str, Any]] = None,
                  wavelength: Optional[np.ndarray] = None) -> None:
        """
        Guarda un cubo de datos en formato FITS.

        Args:
            data: Array 3D de datos
            filename: Nombre del archivo de salida
            header: Header FITS opcional
            wavelength: Array de longitudes de onda opcional
        """
        pass

    @abstractmethod
    def validate_fits_structure(self, filename: Union[str, Path]) -> Dict[str, Any]:
        """
        Valida la estructura de un archivo FITS.

        Args:
            filename: Ruta al archivo FITS

        Returns:
            Diccionario con información de validación
        """
        pass

    @abstractmethod
    def get_extensions_info(self, filename: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Obtiene información sobre las extensiones de un archivo FITS.

        Args:
            filename: Ruta al archivo FITS

        Returns:
            Lista con información de cada extensión
        """
        pass


class ConfigRepositoryInterface(ABC):
    """
    Interfaz para repositorios que manejan configuración.
    Define operaciones para gestión de configuraciones de aplicación.
    """

    @abstractmethod
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Carga configuración desde un archivo.

        Args:
            config_path: Ruta al archivo de configuración (opcional)

        Returns:
            Diccionario con la configuración cargada

        Raises:
            FileNotFoundError: Si el archivo de configuración no existe
            ValueError: Si la configuración no es válida
        """
        pass

    @abstractmethod
    def save_config(self,
                    config: Dict[str, Any],
                    config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Guarda configuración en un archivo.

        Args:
            config: Configuración a guardar
            config_path: Ruta al archivo de configuración (opcional)

        Raises:
            IOError: Si no se puede escribir el archivo
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
    def merge_config(self,
                     default_config: Dict[str, Any],
                     user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combina configuración por defecto con configuración de usuario.

        Args:
            default_config: Configuración por defecto
            user_config: Configuración de usuario

        Returns:
            Configuración combinada con precedencia del usuario
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valida una configuración.

        Args:
            config: Configuración a validar

        Returns:
            Tupla (es_válida, lista_de_errores)
        """
        pass

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Obtiene el esquema de configuración válida.

        Returns:
            Diccionario con el esquema de configuración
        """
        pass


class FilterRepositoryInterface(ABC):
    """
    Interfaz para repositorios que manejan filtros espectrales.
    """

    @abstractmethod
    def load_filter(self, filter_name: str) -> Dict[str, Any]:
        """
        Carga un filtro por nombre.

        Args:
            filter_name: Nombre del filtro

        Returns:
            Diccionario con datos del filtro
        """
        pass

    @abstractmethod
    def list_available_filters(self) -> List[str]:
        """
        Lista todos los filtros disponibles.

        Returns:
            Lista de nombres de filtros
        """
        pass

    @abstractmethod
    def save_filter(self,
                    filter_name: str,
                    filter_data: Dict[str, Any]) -> None:
        """
        Guarda un filtro.

        Args:
            filter_name: Nombre del filtro
            filter_data: Datos del filtro
        """
        pass


class CacheRepositoryInterface(ABC):
    """
    Interfaz para repositorios con capacidades de caché.
    """

    @abstractmethod
    def get_from_cache(self, key: str) -> Optional[Any]:
        """
        Obtiene un elemento del caché.

        Args:
            key: Clave del elemento

        Returns:
            Elemento del caché o None si no existe
        """
        pass

    @abstractmethod
    def put_in_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Almacena un elemento en el caché.

        Args:
            key: Clave del elemento
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos (opcional)
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Limpia todo el caché."""
        pass

    @abstractmethod
    def remove_from_cache(self, key: str) -> bool:
        """
        Elimina un elemento del caché.

        Args:
            key: Clave del elemento

        Returns:
            True si se eliminó, False si no existía
        """
        pass