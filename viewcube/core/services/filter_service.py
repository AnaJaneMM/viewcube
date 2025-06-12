"""
Servicio de filtros para ViewCube.

Este módulo maneja todas las operaciones relacionadas con filtros espectrales,
incluyendo carga, aplicación y procesamiento de bandas de paso.
"""

import os
import glob
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

from ..domain.models.spectrum_data import SpectrumData
from ..domain.models.filter_data import FilterData
from ..domain.models.cube_data import CubeData


class FilterService:
    """
    Servicio para el manejo de filtros espectrales.

    Centraliza todas las operaciones relacionadas con la carga, aplicación
    y procesamiento de filtros espectrales y bandas de paso.
    """

    def __init__(self, filter_directory: str = "filters/"):
        """
        Inicializa el servicio de filtros.

        Args:
            filter_directory: Directorio donde se encuentran los archivos de filtros
        """
        self.filter_directory = Path(filter_directory)
        self._loaded_filters = {}
        self._filter_cache = {}

    def list_available_filters(self) -> List[str]:
        """
        Lista todos los filtros disponibles en el directorio.

        Returns:
            Lista de nombres de filtros disponibles
        """
        if not self.filter_directory.exists():
            return []

        # Buscar archivos de filtro (típicamente .txt o .dat)
        filter_files = []
        for pattern in ['*.txt', '*.dat', '*.filter']:
            filter_files.extend(glob.glob(str(self.filter_directory / pattern)))

        # Extraer nombres base sin extensión
        filter_names = [Path(f).stem for f in filter_files]
        return sorted(filter_names)

    def load_filter(self, filter_name: str, force_reload: bool = False) -> FilterData:
        """
        Carga un filtro desde archivo.

        Args:
            filter_name: Nombre del filtro (sin extensión)
            force_reload: Forzar recarga aunque esté en caché

        Returns:
            Objeto FilterData

        Raises:
            FileNotFoundError: Si el filtro no existe
            ValueError: Si el archivo no tiene el formato correcto
        """
        # Verificar caché
        if filter_name in self._loaded_filters and not force_reload:
            return self._loaded_filters[filter_name]

        # Buscar archivo del filtro
        filter_file = self._find_filter_file(filter_name)
        if filter_file is None:
            raise FileNotFoundError(f'Filtro "{filter_name}" no encontrado en {self.filter_directory}')

        try:
            # Cargar datos del filtro
            wavelength, response = self._load_filter_data(filter_file)

            # Crear objeto FilterData
            filter_data = FilterData(
                wavelength=wavelength,
                response=response,
                name=filter_name,
                meta={'filename': str(filter_file)}
            )

            # Guardar en caché
            self._loaded_filters[filter_name] = filter_data

            return filter_data

        except Exception as e:
            raise ValueError(f'Error cargando filtro "{filter_name}": {e}')

    def _find_filter_file(self, filter_name: str) -> Optional[Path]:
        """
        Busca el archivo correspondiente a un filtro.

        Args:
            filter_name: Nombre del filtro

        Returns:
            Path al archivo del filtro o None si no se encuentra
        """
        # Buscar archivos que contengan el nombre del filtro
        for pattern in ['*.txt', '*.dat', '*.filter']:
            matches = list(self.filter_directory.glob(f'*{filter_name}*{pattern[1:]}'))
            if matches:
                return matches[0]

        # Buscar por nombre exacto
        for extension in ['.txt', '.dat', '.filter']:
            candidate = self.filter_directory / f'{filter_name}{extension}'
            if candidate.exists():
                return candidate

        return None

    def _load_filter_data(self, filter_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga datos de wavelength y response desde archivo.

        Args:
            filter_file: Path al archivo del filtro

        Returns:
            Tupla (wavelength, response)
        """
        try:
            # Intentar carga directa (formato de dos columnas)
            data = np.loadtxt(filter_file)
            if data.shape[1] >= 2:
                return data[:, 0], data[:, 1]
            else:
                raise ValueError("El archivo debe tener al menos 2 columnas")

        except Exception as e:
            # Intentar formatos alternativos
            try:
                # Formato con header o comentarios
                data = np.loadtxt(filter_file, comments=['#', '!', '%'])
                if data.shape[1] >= 2:
                    return data[:, 0], data[:, 1]
                else:
                    raise ValueError("Formato de archivo no reconocido")
            except:
                raise ValueError(f"No se pudo cargar el archivo del filtro: {e}")

    def find_filter_by_name(self, partial_name: str,
                            available_filters: Optional[List[str]] = None) -> Optional[str]:
        """
        Busca un filtro por nombre parcial.

        Args:
            partial_name: Nombre parcial del filtro
            available_filters: Lista de filtros disponibles (opcional)

        Returns:
            Nombre completo del filtro encontrado o None
        """
        if available_filters is None:
            available_filters = self.list_available_filters()

        # Buscar coincidencia exacta
        if partial_name in available_filters:
            return partial_name

        # Buscar coincidencias parciales
        matches = [f for f in available_filters if partial_name.lower() in f.lower()]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # Preferir coincidencias más exactas
            exact_matches = [f for f in matches if partial_name.lower() == f.lower()]
            if exact_matches:
                return exact_matches[0]

            # Preferir coincidencias que empiecen con el nombre parcial
            start_matches = [f for f in matches if f.lower().startswith(partial_name.lower())]
            if start_matches:
                return start_matches[0]

            return matches[0]  # Devolver la primera coincidencia

        return None

    def apply_filter_to_spectrum(self, spectrum: SpectrumData,
                                 filter_data: FilterData,
                                 normalize: bool = True) -> SpectrumData:
        """
        Aplica un filtro a un espectro.

        Args:
            spectrum: Espectro de entrada
            filter_data: Datos del filtro
            normalize: Normalizar por la respuesta del filtro

        Returns:
            Espectro filtrado
        """
        # Aplicar filtro
        filtered_spectrum = filter_data.apply_to_spectrum(spectrum)

        # Normalización opcional
        if normalize:
            # Calcular factor de normalización
            norm_factor = np.trapz(filter_data.response, filter_data.wavelength)
            if norm_factor != 0:
                filtered_spectrum.flux = filtered_spectrum.flux / norm_factor
                if filtered_spectrum.error is not None:
                    filtered_spectrum.error = filtered_spectrum.error / norm_factor

        return filtered_spectrum

    def apply_filter_to_cube(self, cube: CubeData,
                             filter_data: FilterData,
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
        # Interpolar respuesta del filtro a la wavelength del cubo
        filter_response = np.interp(cube.wavelength, filter_data.wavelength, filter_data.response)

        # Aplicar filtro a cada spaxel
        filtered_image = np.zeros((cube.n_y, cube.n_x))

        for y in range(cube.n_y):
            for x in range(cube.n_x):
                spectrum_flux = cube.data[:, y, x]
                # Integrar bajo el filtro
                filtered_flux = np.trapz(spectrum_flux * filter_response, cube.wavelength)

                if normalize:
                    norm_factor = np.trapz(filter_response, cube.wavelength)
                    if norm_factor != 0:
                        filtered_flux /= norm_factor

                filtered_image[y, x] = filtered_flux

        return filtered_image

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
        max_response = np.max(filter_data.response)
        threshold_response = max_response * threshold

        # Encontrar índices donde la respuesta supera el umbral
        above_threshold = filter_data.response >= threshold_response
        valid_indices = np.where(above_threshold)[0]

        if len(valid_indices) == 0:
            return (np.min(filter_data.wavelength), np.max(filter_data.wavelength))

        wave_min = filter_data.wavelength[valid_indices[0]]
        wave_max = filter_data.wavelength[valid_indices[-1]]

        return (wave_min, wave_max)

    def calculate_effective_wavelength(self, filter_data: FilterData) -> float:
        """
        Calcula la longitud de onda efectiva del filtro.

        Args:
            filter_data: Datos del filtro

        Returns:
            Longitud de onda efectiva
        """
        # Método estándar: integral ponderada
        numerator = np.trapz(filter_data.wavelength * filter_data.response, filter_data.wavelength)
        denominator = np.trapz(filter_data.response, filter_data.wavelength)

        if denominator != 0:
            return numerator / denominator
        else:
            # Fallback: wavelength del máximo de respuesta
            max_idx = np.argmax(filter_data.response)
            return filter_data.wavelength[max_idx]

    def calculate_filter_width(self, filter_data: FilterData) -> float:
        """
        Calcula el ancho efectivo del filtro.

        Args:
            filter_data: Datos del filtro

        Returns:
            Ancho efectivo del filtro
        """
        # Método RMS
        eff_wave = self.calculate_effective_wavelength(filter_data)

        numerator = np.trapz((filter_data.wavelength - eff_wave) ** 2 * filter_data.response,
                             filter_data.wavelength)
        denominator = np.trapz(filter_data.response, filter_data.wavelength)

        if denominator != 0:
            return np.sqrt(numerator / denominator)
        else:
            return 0.0

    def create_synthetic_filter(self, wavelength: np.ndarray,
                                center: float,
                                width: float,
                                filter_type: str = 'gaussian',
                                name: Optional[str] = None) -> FilterData:
        """
        Crea un filtro sintético.

        Args:
            wavelength: Array de longitudes de onda
            center: Longitud de onda central
            width: Ancho del filtro
            filter_type: Tipo de filtro ('gaussian', 'tophat', 'triangular')
            name: Nombre del filtro

        Returns:
            Objeto FilterData con el filtro sintético
        """
        if filter_type.lower() == 'gaussian':
            response = np.exp(-0.5 * ((wavelength - center) / (width / 2.355)) ** 2)
        elif filter_type.lower() == 'tophat':
            response = np.where(np.abs(wavelength - center) <= width / 2, 1.0, 0.0)
        elif filter_type.lower() == 'triangular':
            response = np.maximum(0, 1 - np.abs(wavelength - center) / (width / 2))
        else:
            raise ValueError(f"Tipo de filtro no soportado: {filter_type}")

        filter_name = name or f"{filter_type}_{center:.0f}_{width:.0f}"

        return FilterData(
            wavelength=wavelength,
            response=response,
            name=filter_name,
            meta={'type': 'synthetic', 'filter_type': filter_type}
        )

    def convolve_spectrum_with_filter(self, spectrum: SpectrumData,
                                      filter_data: FilterData,
                                      method: str = 'integrate') -> float:
        """
        Convoluciona un espectro con un filtro.

        Args:
            spectrum: Espectro de entrada
            filter_data: Datos del filtro
            method: Método de convolución ('integrate', 'weighted_mean')

        Returns:
            Flujo integrado bajo el filtro
        """
        if method == 'integrate':
            return filter_data.integrate_flux(spectrum)
        elif method == 'weighted_mean':
            # Interpolar respuesta del filtro
            filter_response = np.interp(spectrum.wavelength, filter_data.wavelength, filter_data.response)

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
        shifted_wavelength = filter_data.wavelength + delta_wavelength

        shifted_name = new_name or f"{filter_data.name}_shifted_{delta_wavelength:+.1f}"

        return FilterData(
            wavelength=shifted_wavelength,
            response=filter_data.response.copy(),
            name=shifted_name,
            meta=filter_data.meta.copy()
        )

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
        if extrapolate:
            # Interpolar con extrapolación
            new_response = np.interp(new_wavelength, filter_data.wavelength, filter_data.response)
        else:
            # Interpolar sin extrapolación (valores fuera del rango = 0)
            new_response = np.interp(new_wavelength, filter_data.wavelength, filter_data.response,
                                     left=0.0, right=0.0)

        return FilterData(
            wavelength=new_wavelength,
            response=new_response,
            name=f"{filter_data.name}_interp",
            meta=filter_data.meta.copy()
        )

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
        if not filters:
            raise ValueError("Se requiere al menos un filtro")

        if weights is None:
            weights = [1.0] * len(filters)

        if len(weights) != len(filters):
            raise ValueError("El número de pesos debe coincidir con el número de filtros")

        # Encontrar rango común de wavelength
        wave_min = max(f.wavelength.min() for f in filters)
        wave_max = min(f.wavelength.max() for f in filters)

        # Crear grilla común
        n_points = 1000
        common_wavelength = np.linspace(wave_min, wave_max, n_points)

        # Interpolar todos los filtros a la grilla común
        combined_response = np.zeros_like(common_wavelength)

        for i, (filter_data, weight) in enumerate(zip(filters, weights)):
            interp_response = np.interp(common_wavelength, filter_data.wavelength, filter_data.response)
            combined_response += weight * interp_response

        # Normalizar
        if np.max(combined_response) > 0:
            combined_response /= np.max(combined_response)

        combined_name = name or f"combined_{'_'.join(f.name for f in filters[:3])}"

        return FilterData(
            wavelength=common_wavelength,
            response=combined_response,
            name=combined_name,
            meta={'type': 'combined', 'component_filters': [f.name for f in filters]}
        )

    def get_filter_statistics(self, filter_data: FilterData) -> Dict[str, float]:
        """
        Calcula estadísticas del filtro.

        Args:
            filter_data: Datos del filtro

        Returns:
            Diccionario con estadísticas del filtro
        """
        effective_wave = self.calculate_effective_wavelength(filter_data)
        width = self.calculate_filter_width(filter_data)
        wave_min, wave_max = self.calculate_passband_limits(filter_data)

        return {
            'effective_wavelength': effective_wave,
            'width': width,
            'passband_min': wave_min,
            'passband_max': wave_max,
            'passband_range': wave_max - wave_min,
            'max_response': np.max(filter_data.response),
            'integral': np.trapz(filter_data.response, filter_data.wavelength)
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
        effective_wave = self.calculate_effective_wavelength(filter_data)
        wave_min, wave_max = wave_range

        return wave_min <= effective_wave <= wave_max

    def clear_cache(self) -> None:
        """Limpia la caché de filtros cargados."""
        self._loaded_filters.clear()
        self._filter_cache.clear()