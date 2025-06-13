"""
Servicio de filtros para ViewCube.

Este módulo maneja todas las operaciones relacionadas con filtros espectrales,
incluyendo carga, aplicación y procesamiento de bandas de paso.
"""

import os
import glob
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Protocol
from pathlib import Path
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

from ..domain.models.spectrum_data import SpectrumData
from ..domain.models.filter_data import FilterData
from ..domain.models.cube_data import CubeData

logger = logging.getLogger(__name__)


class FilterType(Enum):
    """Tipos de filtros soportados."""
    GAUSSIAN = "gaussian"
    TOPHAT = "tophat"
    TRIANGULAR = "triangular"
    CUSTOM = "custom"


class FilterLoaderProtocol(Protocol):
    """Protocolo para cargadores de filtros."""

    def load(self, filename: str) -> FilterData:
        """Carga un filtro desde archivo."""
        ...


class FilterValidatorProtocol(Protocol):
    """Protocolo para validadores de filtros."""

    def validate(self, filter_data: FilterData) -> bool:
        """Valida datos de filtro."""
        ...


class FilterProcessorProtocol(Protocol):
    """Protocolo para procesadores de filtros."""

    def process(self, spectrum: SpectrumData, filter_data: FilterData) -> SpectrumData:
        """Procesa un espectro con un filtro."""
        ...


@dataclass
class FilterSearchResult:
    """Resultado de búsqueda de filtros."""
    exact_matches: List[str]
    partial_matches: List[str]
    suggested: Optional[str]


class FilterValidator:
    """Validador especializado para datos de filtro."""

    @staticmethod
    def validate_filter_data(filter_data: FilterData) -> bool:
        """
        Valida que los datos de filtro sean consistentes.

        Args:
            filter_data: Datos del filtro a validar

        Returns:
            True si los datos son válidos

        Raises:
            ValueError: Si los datos son inválidos
        """
        if filter_data.wavelength is None or filter_data.response is None:
            raise ValueError("Filtro debe tener wavelength y response")

        if len(filter_data.wavelength) != len(filter_data.response):
            raise ValueError("Wavelength y response deben tener la misma longitud")

        if len(filter_data.wavelength) < 2:
            raise ValueError("Filtro debe tener al menos 2 puntos")

        if not np.all(np.isfinite(filter_data.wavelength)):
            raise ValueError("Wavelength contiene valores no finitos")

        if not np.all(np.isfinite(filter_data.response)):
            raise ValueError("Response contiene valores no finitos")

        if not np.all(filter_data.response >= 0):
            raise ValueError("Response debe ser no negativa")

        if np.max(filter_data.response) == 0:
            raise ValueError("Response no puede ser toda ceros")

        # Verificar que wavelength sea monótona
        if not np.all(np.diff(filter_data.wavelength) > 0):
            raise ValueError("Wavelength debe ser monótona creciente")

        return True

    @staticmethod
    def validate_filter_file(filename: str) -> bool:
        """
        Valida que un archivo de filtro sea accesible.

        Args:
            filename: Ruta al archivo

        Returns:
            True si el archivo es válido
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Archivo de filtro "{filename}" no existe')

        if not os.path.isfile(filename):
            raise ValueError(f'"{filename}" no es un archivo')

        if os.path.getsize(filename) == 0:
            raise ValueError(f'Archivo "{filename}" está vacío')

        return True


class FilterFileLoader:
    """Cargador especializado para archivos de filtro."""

    def __init__(self, validator: Optional[FilterValidator] = None):
        """
        Inicializa el cargador de filtros.

        Args:
            validator: Validador de filtros (opcional)
        """
        self._validator = validator or FilterValidator()

    def load_filter_from_file(self, filename: str, name: Optional[str] = None) -> FilterData:
        """
        Carga un filtro desde archivo con validación completa.

        Args:
            filename: Ruta al archivo del filtro
            name: Nombre del filtro (opcional)

        Returns:
            Objeto FilterData validado

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato es inválido
        """
        self._validator.validate_filter_file(filename)

        try:
            wavelength, response = self._load_filter_data(filename)

            filter_name = name or Path(filename).stem
            filter_data = FilterData(
                wavelength=wavelength,
                response=response,
                name=filter_name,
                meta={'filename': str(filename), 'source': 'file'}
            )

            self._validator.validate_filter_data(filter_data)

            logger.debug(f"Filtro cargado exitosamente: {filter_name}")
            return filter_data

        except Exception as e:
            logger.error(f"Error cargando filtro desde {filename}: {e}")
            raise ValueError(f'Error cargando filtro "{filename}": {e}')

    def _load_filter_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga datos de wavelength y response desde archivo.

        Args:
            filename: Ruta al archivo

        Returns:
            Tupla (wavelength, response)
        """
        file_path = Path(filename)

        # Intentar diferentes métodos de carga
        loaders = [
            self._load_simple_format,
            self._load_commented_format,
            self._load_csv_format
        ]

        last_error = None
        for loader in loaders:
            try:
                return loader(file_path)
            except Exception as e:
                last_error = e
                continue

        raise ValueError(f"No se pudo cargar el archivo. Último error: {last_error}")

    def _load_simple_format(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Carga formato simple de dos columnas."""
        data = np.loadtxt(file_path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Formato simple requiere al menos 2 columnas")
        return data[:, 0], data[:, 1]

    def _load_commented_format(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Carga formato con comentarios."""
        data = np.loadtxt(file_path, comments=['#', '!', '%'])
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Formato comentado requiere al menos 2 columnas")
        return data[:, 0], data[:, 1]

    def _load_csv_format(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Carga formato CSV."""
        data = np.loadtxt(file_path, delimiter=',')
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Formato CSV requiere al menos 2 columnas")
        return data[:, 0], data[:, 1]


class FilterDiscovery:
    """Servicio para descubrir y buscar filtros."""

    def __init__(self, filter_directory: Path):
        """
        Inicializa el servicio de descubrimiento.

        Args:
            filter_directory: Directorio de filtros
        """
        self.filter_directory = filter_directory
        self._supported_extensions = ['.txt', '.dat', '.filter', '.csv']

    def list_available_filters(self) -> List[str]:
        """
        Lista todos los filtros disponibles en el directorio.

        Returns:
            Lista de nombres de filtros disponibles
        """
        if not self.filter_directory.exists():
            logger.warning(f"Directorio de filtros no existe: {self.filter_directory}")
            return []

        filter_files = []
        for ext in self._supported_extensions:
            pattern = f'*{ext}'
            filter_files.extend(self.filter_directory.glob(pattern))

        # Extraer nombres base sin extensión y eliminar duplicados
        filter_names = list(set(f.stem for f in filter_files))
        return sorted(filter_names)

    def find_filter_files(self, filter_name: str) -> List[Path]:
        """
        Busca archivos que correspondan a un nombre de filtro.

        Args:
            filter_name: Nombre del filtro a buscar

        Returns:
            Lista de archivos encontrados
        """
        candidates = []

        # Buscar coincidencias exactas
        for ext in self._supported_extensions:
            exact_match = self.filter_directory / f'{filter_name}{ext}'
            if exact_match.exists():
                candidates.append(exact_match)

        # Buscar coincidencias parciales si no hay exactas
        if not candidates:
            for ext in self._supported_extensions:
                pattern = f'*{filter_name}*{ext}'
                candidates.extend(self.filter_directory.glob(pattern))

        return list(set(candidates))  # Eliminar duplicados

    def search_filters(self, partial_name: str,
                       available_filters: Optional[List[str]] = None) -> FilterSearchResult:
        """
        Busca filtros por nombre parcial con resultados estructurados.

        Args:
            partial_name: Nombre parcial del filtro
            available_filters: Lista de filtros disponibles (opcional)

        Returns:
            Resultado de búsqueda estructurado
        """
        if available_filters is None:
            available_filters = self.list_available_filters()

        partial_lower = partial_name.lower()

        # Coincidencias exactas
        exact_matches = [f for f in available_filters if f.lower() == partial_lower]

        # Coincidencias parciales
        partial_matches = [f for f in available_filters
                           if partial_lower in f.lower() and f not in exact_matches]

        # Sugerir mejor coincidencia
        suggested = None
        if exact_matches:
            suggested = exact_matches[0]
        elif partial_matches:
            # Preferir coincidencias que empiecen con el nombre parcial
            start_matches = [f for f in partial_matches
                             if f.lower().startswith(partial_lower)]
            suggested = start_matches[0] if start_matches else partial_matches[0]

        return FilterSearchResult(
            exact_matches=exact_matches,
            partial_matches=partial_matches,
            suggested=suggested
        )


class FilterCalculator:
    """Calculadora especializada para propiedades de filtros."""

    @staticmethod
    def calculate_effective_wavelength(filter_data: FilterData) -> float:
        """
        Calcula la longitud de onda efectiva del filtro.

        Args:
            filter_data: Datos del filtro

        Returns:
            Longitud de onda efectiva
        """
        numerator = np.trapz(
            filter_data.wavelength * filter_data.response,
            filter_data.wavelength
        )
        denominator = np.trapz(filter_data.response, filter_data.wavelength)

        if denominator != 0:
            return numerator / denominator
        else:
            # Fallback: wavelength del máximo de respuesta
            max_idx = np.argmax(filter_data.response)
            return filter_data.wavelength[max_idx]

    @staticmethod
    def calculate_filter_width(filter_data: FilterData) -> float:
        """
        Calcula el ancho efectivo del filtro (RMS).

        Args:
            filter_data: Datos del filtro

        Returns:
            Ancho efectivo del filtro
        """
        eff_wave = FilterCalculator.calculate_effective_wavelength(filter_data)

        numerator = np.trapz(
            (filter_data.wavelength - eff_wave) ** 2 * filter_data.response,
            filter_data.wavelength
        )
        denominator = np.trapz(filter_data.response, filter_data.wavelength)

        if denominator != 0:
            return np.sqrt(numerator / denominator)
        else:
            return 0.0

    @staticmethod
    def calculate_passband_limits(filter_data: FilterData,
                                  threshold: float = 0.1) -> Tuple[float, float]:
        """
        Calcula los límites efectivos de la banda de paso.

        Args:
            filter_data: Datos del filtro
            threshold: Umbral de respuesta (fracción del máximo)

        Returns:
            Tupla (wavelength_min, wavelength_max)
        """
        if not (0 < threshold <= 1):
            raise ValueError("Threshold debe estar entre 0 y 1")

        max_response = np.max(filter_data.response)
        threshold_response = max_response * threshold

        above_threshold = filter_data.response >= threshold_response
        valid_indices = np.where(above_threshold)[0]

        if len(valid_indices) == 0:
            return (np.min(filter_data.wavelength), np.max(filter_data.wavelength))

        wave_min = filter_data.wavelength[valid_indices[0]]
        wave_max = filter_data.wavelength[valid_indices[-1]]

        return (wave_min, wave_max)

    @staticmethod
    def calculate_filter_integral(filter_data: FilterData) -> float:
        """
        Calcula la integral del filtro.

        Args:
            filter_data: Datos del filtro

        Returns:
            Valor de la integral
        """
        return np.trapz(filter_data.response, filter_data.wavelength)


class FilterProcessor:
    """Procesador especializado para aplicar filtros a espectros."""

    def __init__(self, calculator: Optional[FilterCalculator] = None):
        """
        Inicializa el procesador de filtros.

        Args:
            calculator: Calculadora de filtros (opcional)
        """
        self._calculator = calculator or FilterCalculator()

    def apply_filter_to_spectrum(self, spectrum: SpectrumData,
                                 filter_data: FilterData,
                                 normalize: bool = True) -> SpectrumData:
        """
        Aplica un filtro a un espectro con validación.

        Args:
            spectrum: Espectro de entrada
            filter_data: Datos del filtro
            normalize: Normalizar por la respuesta del filtro

        Returns:
            Espectro filtrado
        """
        # Validar entrada
        if not hasattr(spectrum, 'wavelength') or not hasattr(spectrum, 'flux'):
            raise ValueError("Spectrum debe tener wavelength y flux")

        # Aplicar filtro usando el método del FilterData
        filtered_spectrum = filter_data.apply_to_spectrum(spectrum)

        # Normalización opcional
        if normalize:
            norm_factor = self._calculator.calculate_filter_integral(filter_data)
            if norm_factor != 0:
                filtered_spectrum.flux = filtered_spectrum.flux / norm_factor
                if filtered_spectrum.error is not None:
                    filtered_spectrum.error = filtered_spectrum.error / norm_factor

        return filtered_spectrum

    def apply_filter_to_cube(self, cube: CubeData,
                             filter_data: FilterData,
                             normalize: bool = True) -> np.ndarray:
        """
        Aplica un filtro a un cubo espectral completo con optimización.

        Args:
            cube: Cubo de datos
            filter_data: Datos del filtro
            normalize: Normalizar por la respuesta del filtro

        Returns:
            Imagen filtrada (2D array)
        """
        # Validar entrada
        if not hasattr(cube, 'wavelength') or not hasattr(cube, 'data'):
            raise ValueError("Cube debe tener wavelength y data")

        if cube.data.ndim != 3:
            raise ValueError("Cube data debe ser 3D")

        # Interpolar respuesta del filtro a la wavelength del cubo
        filter_response = self._interpolate_filter_response(
            cube.wavelength, filter_data
        )

        # Calcular factor de normalización una vez
        norm_factor = 1.0
        if normalize:
            norm_factor = np.trapz(filter_response, cube.wavelength)
            if norm_factor == 0:
                norm_factor = 1.0

        # Aplicar filtro vectorizado
        filtered_image = self._apply_filter_vectorized(
            cube.data, filter_response, cube.wavelength, norm_factor
        )

        return filtered_image

    def _interpolate_filter_response(self, target_wavelength: np.ndarray,
                                     filter_data: FilterData) -> np.ndarray:
        """
        Interpola la respuesta del filtro a nueva grilla de wavelength.

        Args:
            target_wavelength: Grilla objetivo
            filter_data: Datos del filtro

        Returns:
            Respuesta interpolada
        """
        return np.interp(
            target_wavelength,
            filter_data.wavelength,
            filter_data.response,
            left=0.0,
            right=0.0
        )

    def _apply_filter_vectorized(self, cube_data: np.ndarray,
                                 filter_response: np.ndarray,
                                 wavelength: np.ndarray,
                                 norm_factor: float) -> np.ndarray:
        """
        Aplica filtro de forma vectorizada para mejor rendimiento.

        Args:
            cube_data: Datos del cubo (lambda, y, x)
            filter_response: Respuesta del filtro interpolada
            wavelength: Array de wavelength
            norm_factor: Factor de normalización

        Returns:
            Imagen filtrada
        """
        # Multiplicar por respuesta del filtro
        weighted_data = cube_data * filter_response[:, np.newaxis, np.newaxis]

        # Integrar sobre wavelength
        filtered_image = np.trapz(weighted_data, wavelength, axis=0)

        # Normalizar si es necesario
        if norm_factor != 1.0:
            filtered_image /= norm_factor

        return filtered_image


class SyntheticFilterFactory:
    """Factory para crear filtros sintéticos."""

    @staticmethod
    def create_gaussian_filter(wavelength: np.ndarray, center: float,
                               width: float, name: Optional[str] = None) -> FilterData:
        """
        Crea un filtro gaussiano.

        Args:
            wavelength: Array de longitudes de onda
            center: Longitud de onda central
            width: Ancho FWHM del filtro
            name: Nombre del filtro

        Returns:
            FilterData con filtro gaussiano
        """
        sigma = width / 2.355  # Convertir FWHM a sigma
        response = np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)

        filter_name = name or f"gaussian_{center:.0f}_{width:.0f}"

        return FilterData(
            wavelength=wavelength,
            response=response,
            name=filter_name,
            meta={
                'type': 'synthetic',
                'filter_type': FilterType.GAUSSIAN.value,
                'center': center,
                'width': width
            }
        )

    @staticmethod
    def create_tophat_filter(wavelength: np.ndarray, center: float,
                             width: float, name: Optional[str] = None) -> FilterData:
        """
        Crea un filtro tophat (rectangular).

        Args:
            wavelength: Array de longitudes de onda
            center: Longitud de onda central
            width: Ancho del filtro
            name: Nombre del filtro

        Returns:
            FilterData con filtro tophat
        """
        response = np.where(np.abs(wavelength - center) <= width / 2, 1.0, 0.0)

        filter_name = name or f"tophat_{center:.0f}_{width:.0f}"

        return FilterData(
            wavelength=wavelength,
            response=response,
            name=filter_name,
            meta={
                'type': 'synthetic',
                'filter_type': FilterType.TOPHAT.value,
                'center': center,
                'width': width
            }
        )

    @staticmethod
    def create_triangular_filter(wavelength: np.ndarray, center: float,
                                 width: float, name: Optional[str] = None) -> FilterData:
        """
        Crea un filtro triangular.

        Args:
            wavelength: Array de longitudes de onda
            center: Longitud de onda central
            width: Ancho base del triángulo
            name: Nombre del filtro

        Returns:
            FilterData con filtro triangular
        """
        response = np.maximum(0, 1 - np.abs(wavelength - center) / (width / 2))

        filter_name = name or f"triangular_{center:.0f}_{width:.0f}"

        return FilterData(
            wavelength=wavelength,
            response=response,
            name=filter_name,
            meta={
                'type': 'synthetic',
                'filter_type': FilterType.TRIANGULAR.value,
                'center': center,
                'width': width
            }
        )


class FilterModifier:
    """Modificador para transformar filtros existentes."""

    @staticmethod
    def shift_filter(filter_data: FilterData, delta_wavelength: float,
                     new_name: Optional[str] = None) -> FilterData:
        """
        Desplaza un filtro en longitud de onda.

        Args:
            filter_data: Filtro original
            delta_wavelength: Desplazamiento en longitud de onda
            new_name: Nuevo nombre para el filtro desplazado

        Returns:
            Nuevo FilterData con el filtro desplazado
        """
        shifted_wavelength = filter_data.wavelength + delta_wavelength
        shifted_name = new_name or f"{filter_data.name}_shifted_{delta_wavelength:+.1f}"

        meta = filter_data.meta.copy()
        meta['shift_applied'] = delta_wavelength
        meta['original_name'] = filter_data.name

        return FilterData(
            wavelength=shifted_wavelength,
            response=filter_data.response.copy(),
            name=shifted_name,
            meta=meta
        )

    @staticmethod
    def interpolate_filter(filter_data: FilterData, new_wavelength: np.ndarray,
                           extrapolate: bool = False) -> FilterData:
        """
        Interpola un filtro a una nueva grilla de wavelength.

        Args:
            filter_data: Filtro original
            new_wavelength: Nueva grilla de wavelength
            extrapolate: Permitir extrapolación fuera del rango original

        Returns:
            FilterData interpolado
        """
        if not isinstance(new_wavelength, np.ndarray):
            new_wavelength = np.array(new_wavelength)

        if extrapolate:
            new_response = np.interp(
                new_wavelength, filter_data.wavelength, filter_data.response
            )
        else:
            new_response = np.interp(
                new_wavelength, filter_data.wavelength, filter_data.response,
                left=0.0, right=0.0
            )

        meta = filter_data.meta.copy()
        meta['interpolated'] = True
        meta['original_points'] = len(filter_data.wavelength)
        meta['new_points'] = len(new_wavelength)

        return FilterData(
            wavelength=new_wavelength,
            response=new_response,
            name=f"{filter_data.name}_interp",
            meta=meta
        )

    @staticmethod
    def combine_filters(filters: List[FilterData], weights: Optional[List[float]] = None,
                        name: Optional[str] = None) -> FilterData:
        """
        Combina múltiples filtros con pesos opcionales.

        Args:
            filters: Lista de filtros a combinar
            weights: Pesos para cada filtro (opcional)
            name: Nombre del filtro combinado

        Returns:
            FilterData combinado
        """
        if not filters:
            raise ValueError("Se requiere al menos un filtro")

        if weights is None:
            weights = [1.0] * len(filters)

        if len(weights) != len(filters):
            raise ValueError("El número de pesos debe coincidir con el número de filtros")

        # Encontrar rango común de wavelength
        wave_min = max(f.wavelength.min() for f in filters)
        wave_max = min(f.wavelength.max() for f in filters)

        if wave_min >= wave_max:
            raise ValueError("Los filtros no tienen rango de wavelength común")

        # Crear grilla común de alta resolución
        n_points = max(1000, max(len(f.wavelength) for f in filters))
        common_wavelength = np.linspace(wave_min, wave_max, n_points)

        # Interpolar todos los filtros a la grilla común
        combined_response = np.zeros_like(common_wavelength)

        for filter_data, weight in zip(filters, weights):
            interp_response = np.interp(
                common_wavelength, filter_data.wavelength, filter_data.response
            )
            combined_response += weight * interp_response

        # Normalizar al máximo
        max_response = np.max(combined_response)
        if max_response > 0:
            combined_response /= max_response

        combined_name = name or f"combined_{'_'.join(f.name for f in filters[:3])}"

        return FilterData(
            wavelength=common_wavelength,
            response=combined_response,
            name=combined_name,
            meta={
                'type': 'combined',
                'component_filters': [f.name for f in filters],
                'weights': weights
            }
        )


class FilterService:
    """
    Servicio principal de filtros completamente refactorizado.

    Esta implementación aplica principios SOLID, reduce la complejidad ciclomática
    y mejora la separación de responsabilidades manteniendo toda la funcionalidad original.
    """

    def __init__(self, filter_directory: str = "filters/"):
        """
        Inicializa el servicio de filtros con arquitectura modular.

        Args:
            filter_directory: Directorio donde se encuentran los archivos de filtros
        """
        self.filter_directory = Path(filter_directory)

        # Componentes especializados
        self._validator = FilterValidator()
        self._file_loader = FilterFileLoader(self._validator)
        self._discovery = FilterDiscovery(self.filter_directory)
        self._calculator = FilterCalculator()
        self._processor = FilterProcessor(self._calculator)
        self._synthetic_factory = SyntheticFilterFactory()
        self._modifier = FilterModifier()

        # Caché de filtros cargados
        self._loaded_filters = {}
        self._filter_cache = {}

        logger.info(f"FilterService inicializado con directorio: {self.filter_directory}")

    # === MÉTODOS PÚBLICOS DE LA INTERFAZ ORIGINAL ===

    def list_available_filters(self) -> List[str]:
        """
        Lista todos los filtros disponibles en el directorio.

        Returns:
            Lista de nombres de filtros disponibles
        """
        return self._discovery.list_available_filters()

    def load_filter(self, filter_name: str, force_reload: bool = False) -> FilterData:
        """
        Carga un filtro desde archivo con validación mejorada.

        Args:
            filter_name: Nombre del filtro (sin extensión)
            force_reload: Forzar recarga aunque esté en caché

        Returns:
            Objeto FilterData validado

        Raises:
            FileNotFoundError: Si el filtro no existe
            ValueError: Si el archivo no tiene el formato correcto
        """
        # Verificar caché
        if filter_name in self._loaded_filters and not force_reload:
            logger.debug(f"Filtro {filter_name} cargado desde caché")
            return self._loaded_filters[filter_name]

        # Buscar archivo del filtro
        filter_files = self._discovery.find_filter_files(filter_name)

        if not filter_files:
            raise FileNotFoundError(
                f'Filtro "{filter_name}" no encontrado en {self.filter_directory}'
            )

        # Usar el primer archivo encontrado
        filter_file = filter_files[0]

        try:
            filter_data = self._file_loader.load_filter_from_file(
                str(filter_file), filter_name
            )

            # Guardar en caché
            self._loaded_filters[filter_name] = filter_data

            logger.info(f"Filtro {filter_name} cargado exitosamente")
            return filter_data

        except Exception as e:
            logger.error(f"Error cargando filtro {filter_name}: {e}")
            raise

    def find_filter_by_name(self, partial_name: str,
                            available_filters: Optional[List[str]] = None) -> Optional[str]:
        """
        Busca un filtro por nombre parcial con mejores heurísticas.

        Args:
            partial_name: Nombre parcial del filtro
            available_filters: Lista de filtros disponibles (opcional)

        Returns:
            Nombre completo del filtro encontrado o None
        """
        search_result = self._discovery.search_filters(partial_name, available_filters)
        return search_result.suggested

    def apply_filter_to_spectrum(self, spectrum: SpectrumData,
                                 filter_data: FilterData,
                                 normalize: bool = True) -> SpectrumData:
        """
        Aplica un filtro a un espectro con validación.

        Args:
            spectrum: Espectro de entrada
            filter_data: Datos del filtro
            normalize: Normalizar por la respuesta del filtro

        Returns:
            Espectro filtrado validado
        """
        return self._processor.apply_filter_to_spectrum(spectrum, filter_data, normalize)

    def apply_filter_to_cube(self, cube: CubeData,
                             filter_data: FilterData,
                             normalize: bool = True) -> np.ndarray:
        """
        Aplica un filtro a un cubo espectral completo con optimización.

        Args:
            cube: Cubo de datos
            filter_data: Datos del filtro
            normalize: Normalizar por la respuesta del filtro

        Returns:
            Imagen filtrada (2D array)
        """
        return self._processor.apply_filter_to_cube(cube, filter_data, normalize)

    def calculate_passband_limits(self, filter_data: FilterData,
                                  threshold: float = 0.1) -> Tuple[float, float]:
        """
        Calcula los límites efectivos de la banda de paso.

        Args:
            filter_data: Datos del filtro
            threshold: Umbral de respuesta (fracción del máximo)

        Returns:
            Tupla (wavelength_min, wavelength_max)
        """
        return self._calculator.calculate_passband_limits(filter_data, threshold)

    def calculate_effective_wavelength(self, filter_data: FilterData) -> float:
        """
        Calcula la longitud de onda efectiva del filtro.

        Args:
            filter_data: Datos del filtro

        Returns:
            Longitud de onda efectiva
        """
        return self._calculator.calculate_effective_wavelength(filter_data)

    def calculate_filter_width(self, filter_data: FilterData) -> float:
        """
        Calcula el ancho efectivo del filtro.

        Args:
            filter_data: Datos del filtro

        Returns:
            Ancho efectivo del filtro
        """
        return self._calculator.calculate_filter_width(filter_data)

    def create_synthetic_filter(self, wavelength: np.ndarray,
                                center: float,
                                width: float,
                                filter_type: str = 'gaussian',
                                name: Optional[str] = None) -> FilterData:
        """
        Crea un filtro sintético con validación de tipo.

        Args:
            wavelength: Array de longitudes de onda
            center: Longitud de onda central
            width: Ancho del filtro
            filter_type: Tipo de filtro ('gaussian', 'tophat', 'triangular')
            name: Nombre del filtro

        Returns:
            Objeto FilterData con el filtro sintético

        Raises:
            ValueError: Si el tipo de filtro no es soportado
        """
        if not isinstance(wavelength, np.ndarray):
            wavelength = np.array(wavelength)

        filter_type_lower = filter_type.lower()

        if filter_type_lower == FilterType.GAUSSIAN.value:
            return self._synthetic_factory.create_gaussian_filter(
                wavelength, center, width, name
            )
        elif filter_type_lower == FilterType.TOPHAT.value:
            return self._synthetic_factory.create_tophat_filter(
                wavelength, center, width, name
            )
        elif filter_type_lower == FilterType.TRIANGULAR.value:
            return self._synthetic_factory.create_triangular_filter(
                wavelength, center, width, name
            )
        else:
            raise ValueError(f"Tipo de filtro no soportado: {filter_type}")

    def convolve_spectrum_with_filter(self, spectrum: SpectrumData,
                                      filter_data: FilterData,
                                      method: str = 'integrate') -> float:
        """
        Convoluciona un espectro con un filtro con validación de método.

        Args:
            spectrum: Espectro de entrada
            filter_data: Datos del filtro
            method: Método de convolución ('integrate', 'weighted_mean')

        Returns:
            Flujo integrado bajo el filtro

        Raises:
            ValueError: Si el método no es soportado
        """
        if method == 'integrate':
            return filter_data.integrate_flux(spectrum)
        elif method == 'weighted_mean':
            # Interpolar respuesta del filtro
            filter_response = np.interp(
                spectrum.wavelength, filter_data.wavelength, filter_data.response
            )
            # Media ponderada
            weights = filter_response
            weighted_flux = np.average(spectrum.flux, weights=weights)
            return weighted_flux
        else:
            raise ValueError(f"Método no soportado: {method}")

    def shift_filter(self, filter_data: FilterData,
                     delta_wavelength: float,
                     new_name: Optional[str] = None) -> FilterData:
        """
        Desplaza un filtro en longitud de onda.

        Args:
            filter_data: Filtro original
            delta_wavelength: Desplazamiento en longitud de onda
            new_name: Nuevo nombre para el filtro desplazado

        Returns:
            Nuevo FilterData con el filtro desplazado
        """
        return self._modifier.shift_filter(filter_data, delta_wavelength, new_name)

    def interpolate_filter(self, filter_data: FilterData,
                           new_wavelength: np.ndarray,
                           extrapolate: bool = False) -> FilterData:
        """
        Interpola un filtro a una nueva grilla de wavelength.

        Args:
            filter_data: Filtro original
            new_wavelength: Nueva grilla de wavelength
            extrapolate: Permitir extrapolación fuera del rango original

        Returns:
            FilterData interpolado
        """
        return self._modifier.interpolate_filter(filter_data, new_wavelength, extrapolate)

    def combine_filters(self, filters: List[FilterData],
                        weights: Optional[List[float]] = None,
                        name: Optional[str] = None) -> FilterData:
        """
        Combina múltiples filtros.

        Args:
            filters: Lista de filtros a combinar
            weights: Pesos para cada filtro (opcional)
            name: Nombre del filtro combinado

        Returns:
            FilterData combinado
        """
        return self._modifier.combine_filters(filters, weights, name)

    def get_filter_statistics(self, filter_data: FilterData) -> Dict[str, float]:
        """
        Calcula estadísticas completas del filtro.

        Args:
            filter_data: Datos del filtro

        Returns:
            Diccionario con estadísticas del filtro
        """
        effective_wave = self._calculator.calculate_effective_wavelength(filter_data)
        width = self._calculator.calculate_filter_width(filter_data)
        wave_min, wave_max = self._calculator.calculate_passband_limits(filter_data)
        integral = self._calculator.calculate_filter_integral(filter_data)

        return {
            'effective_wavelength': effective_wave,
            'width': width,
            'passband_min': wave_min,
            'passband_max': wave_max,
            'passband_range': wave_max - wave_min,
            'max_response': float(np.max(filter_data.response)),
            'integral': integral,
            'num_points': len(filter_data.wavelength)
        }

    def center_filter_in_range(self, filter_data: FilterData,
                               wave_range: Tuple[float, float]) -> bool:
        """
        Verifica si el filtro está centrado en el rango de wavelength dado.

        Args:
            filter_data: Datos del filtro
            wave_range: Tupla (wave_min, wave_max)

        Returns:
            True si el filtro está dentro del rango
        """
        effective_wave = self._calculator.calculate_effective_wavelength(filter_data)
        wave_min, wave_max = wave_range
        return wave_min <= effective_wave <= wave_max

    # === MÉTODOS ADICIONALES MEJORADOS ===

    def validate_filter(self, filter_data: FilterData) -> bool:
        """
        Valida un filtro externamente.

        Args:
            filter_data: Datos del filtro

        Returns:
            True si el filtro es válido
        """
        return self._validator.validate_filter_data(filter_data)

    def search_filters_advanced(self, partial_name: str) -> FilterSearchResult:
        """
        Búsqueda avanzada de filtros con resultados estructurados.

        Args:
            partial_name: Nombre parcial del filtro

        Returns:
            Resultado de búsqueda estructurado
        """
        return self._discovery.search_filters(partial_name)

    def get_loaded_filters(self) -> List[str]:
        """
        Obtiene la lista de filtros cargados en caché.

        Returns:
            Lista de nombres de filtros cargados
        """
        return list(self._loaded_filters.keys())

    def clear_cache(self) -> None:
        """Limpia la caché de filtros cargados."""
        self._loaded_filters.clear()
        self._filter_cache.clear()
        logger.info("Caché de filtros limpiada")

    def get_cache_info(self) -> Dict[str, int]:
        """
        Obtiene información sobre el uso de caché.

        Returns:
            Diccionario con estadísticas de caché
        """
        return {
            'loaded_filters': len(self._loaded_filters),
            'cache_entries': len(self._filter_cache),
            'available_filters': len(self.list_available_filters())
        }