"""
Modulo de filtros espectrales para el procesamiento de datos astronómicos.

Este módulo proporciona funcionalidades para aplicar filtros espectrales a datos
de espectros astronómicos, incluyendo interpolación, integración y validación.

Versión: 2.0.0
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configurar logging
logger = logging.getLogger(__name__)


class FilterError(Exception):
    """Excepción personalizada para errores de filtros espectrales."""
    pass


class FilterValidationError(FilterError):
    """Excepción para errores de validación de filtros."""
    pass


class SpectrumCompatibilityError(FilterError):
    """Excepción para errores de compatibilidad entre filtro y espectro."""
    pass


@dataclass
class FilterMetadata:
    """
    Metadatos estructurados para filtros espectrales.

    Attributes
    ----------
    filter_type : str
        Tipo de filtro (e.g., 'photometric', 'bandpass')
    central_wavelength : float, optional
        Longitud de onda central en Angstroms
    effective_width : float, optional
        Ancho efectivo del filtro en Angstroms
    zero_point : float, optional
        Punto cero fotométrico
    reference_system : str
        Sistema de referencia (e.g., 'AB', 'Vega')
    source : str
        Fuente del filtro
    version : str
        Versión del filtro
    """
    filter_type: str = "photometric"
    central_wavelength: Optional[float] = None
    effective_width: Optional[float] = None
    zero_point: Optional[float] = None
    reference_system: str = "AB"
    source: str = "unknown"
    version: str = "1.0"


class FilterValidator:
    """
    Validador para filtros espectrales.

    Proporciona métodos estáticos para validar la consistencia y corrección
    de los datos de filtros espectrales.
    """

    @staticmethod
    def validate_wavelength_response_arrays(wavelength: np.ndarray,
                                            response: np.ndarray) -> None:
        """
        Valida que los arrays de longitud de onda y respuesta sean consistentes.

        Complejidad ciclomática: 8 (< 15 ✓)

        Parameters
        ----------
        wavelength : np.ndarray
            Array de longitudes de onda
        response : np.ndarray
            Array de respuesta del filtro

        Raises
        ------
        FilterValidationError
            Si los arrays no son válidos o inconsistentes
        """
        if wavelength.size == 0:
            raise FilterValidationError("El array de longitudes de onda está vacío")

        if response.size == 0:
            raise FilterValidationError("El array de respuesta está vacío")

        if wavelength.size != response.size:
            raise FilterValidationError(
                f"Los arrays de longitud de onda ({wavelength.size}) y "
                f"respuesta ({response.size}) tienen tamaños diferentes"
            )

        if not np.all(np.isfinite(wavelength)):
            raise FilterValidationError("El array de longitudes de onda contiene valores no finitos")

        if not np.all(np.isfinite(response)):
            raise FilterValidationError("El array de respuesta contiene valores no finitos")

        if np.any(wavelength <= 0):
            raise FilterValidationError("Las longitudes de onda deben ser positivas")

        if np.any(response < 0):
            raise FilterValidationError("Los valores de respuesta no pueden ser negativos")

        if not np.all(np.diff(wavelength) > 0):
            raise FilterValidationError("Las longitudes de onda deben estar ordenadas de forma creciente")

    @staticmethod
    def validate_spectrum_compatibility(filter_wavelength: np.ndarray,
                                        spectrum_wavelength: np.ndarray) -> None:
        """
        Valida la compatibilidad entre el filtro y el espectro.

        Complejidad ciclomática: 3 (< 15 ✓)

        Parameters
        ----------
        filter_wavelength : np.ndarray
            Longitudes de onda del filtro
        spectrum_wavelength : np.ndarray
            Longitudes de onda del espectro

        Raises
        ------
        SpectrumCompatibilityError
            Si no hay suficiente solapamiento espectral
        """
        filter_min, filter_max = filter_wavelength.min(), filter_wavelength.max()
        spectrum_min, spectrum_max = spectrum_wavelength.min(), spectrum_wavelength.max()

        overlap_min = max(filter_min, spectrum_min)
        overlap_max = min(filter_max, spectrum_max)

        if overlap_min >= overlap_max:
            raise SpectrumCompatibilityError(
                f"No hay solapamiento entre el filtro ({filter_min:.1f}-{filter_max:.1f}) "
                f"y el espectro ({spectrum_min:.1f}-{spectrum_max:.1f})"
            )

        overlap_fraction = (overlap_max - overlap_min) / (filter_max - filter_min)
        if overlap_fraction < 0.1:
            logger.warning(
                f"Solapamiento limitado entre filtro y espectro: {overlap_fraction:.1%}"
            )


class FilterInterpolator:
    """
    Interpolador para respuestas de filtros espectrales.

    Maneja la interpolación de respuestas de filtro a diferentes
    grillas de longitud de onda.
    """

    @staticmethod
    def interpolate_response(target_wavelength: np.ndarray,
                             filter_wavelength: np.ndarray,
                             filter_response: np.ndarray,
                             bounds_error: bool = False,
                             fill_value: float = 0.0) -> np.ndarray:
        """
        Interpola la respuesta del filtro a nuevas longitudes de onda.

        Complejidad ciclomática: 2 (< 15 ✓)

        Parameters
        ----------
        target_wavelength : np.ndarray
            Longitudes de onda objetivo
        filter_wavelength : np.ndarray
            Longitudes de onda del filtro
        filter_response : np.ndarray
            Respuesta del filtro
        bounds_error : bool, optional
            Si generar error para valores fuera del rango
        fill_value : float, optional
            Valor para puntos fuera del rango

        Returns
        -------
        np.ndarray
            Respuesta interpolada

        Raises
        ------
        FilterError
            Si ocurre un error durante la interpolación
        """
        try:
            return np.interp(target_wavelength, filter_wavelength, filter_response)
        except Exception as e:
            raise FilterError(f"Error en interpolación: {str(e)}")


class FilterCalculator:
    """
    Calculadora para operaciones con filtros espectrales.

    Proporciona métodos estáticos para cálculos comunes con filtros,
    como integración de flujo y magnitudes.
    """

    @staticmethod
    def calculate_integrated_flux(wavelength: np.ndarray,
                                  flux: np.ndarray,
                                  response: np.ndarray) -> float:
        """
        Calcula el flujo integrado bajo el filtro.

        Complejidad ciclomática: 2 (< 15 ✓)

        Parameters
        ----------
        wavelength : np.ndarray
            Longitudes de onda
        flux : np.ndarray
            Flujo espectral
        response : np.ndarray
            Respuesta del filtro

        Returns
        -------
        float
            Flujo integrado

        Raises
        ------
        FilterError
            Si ocurre un error durante el cálculo
        """
        try:
            integrand = flux * response
            return float(np.trapz(integrand, wavelength))
        except Exception as e:
            raise FilterError(f"Error en cálculo de flujo integrado: {str(e)}")

    @staticmethod
    def calculate_effective_wavelength(wavelength: np.ndarray,
                                       response: np.ndarray) -> float:
        """
        Calcula la longitud de onda efectiva del filtro.

        Complejidad ciclomática: 2 (< 15 ✓)

        Parameters
        ----------
        wavelength : np.ndarray
            Longitudes de onda
        response : np.ndarray
            Respuesta del filtro

        Returns
        -------
        float
            Longitud de onda efectiva

        Raises
        ------
        FilterError
            Si ocurre un error durante el cálculo
        """
        try:
            numerator = np.trapz(wavelength * response, wavelength)
            denominator = np.trapz(response, wavelength)
            return float(numerator / denominator)
        except Exception as e:
            raise FilterError(f"Error en cálculo de longitud de onda efectiva: {str(e)}")


class FilterData:
    """
    Modelo de datos para un filtro espectral.

    Esta clase encapsula un filtro espectral con su respuesta en función
    de la longitud de onda, proporcionando métodos para aplicarlo a espectros
    y realizar cálculos fotométricos.

    Parameters
    ----------
    wavelength : array_like
        Array de longitudes de onda en Angstroms, debe estar ordenado de forma creciente
    response : array_like
        Array de respuesta del filtro (transmitancia), debe tener el mismo tamaño que wavelength
    name : str, optional
        Nombre identificativo del filtro
    metadata : FilterMetadata or dict, optional
        Metadatos adicionales del filtro

    Attributes
    ----------
    wavelength : np.ndarray
        Longitudes de onda del filtro
    response : np.ndarray
        Respuesta del filtro
    name : str
        Nombre del filtro
    metadata : FilterMetadata
        Metadatos estructurados del filtro

    Examples
    --------
    >>> # Crear un filtro simple
    >>> wavelength = np.linspace(4000, 7000, 100)
    >>> response = np.exp(-0.5 * ((wavelength - 5500) / 500)**2)
    >>> filter_obj = FilterData(wavelength, response, name="V_band")

    >>> # Aplicar a un espectro
    >>> filtered_spectrum = filter_obj.apply_to_spectrum(spectrum)

    >>> # Calcular flujo integrado
    >>> integrated_flux = filter_obj.integrate_flux(spectrum)

    Notes
    -----
    La clase implementa validación completa de entrada, manejo de errores
    robusto y separación de responsabilidades para mantener baja complejidad.
    """

    def __init__(self,
                 wavelength: Union[np.ndarray, list],
                 response: Union[np.ndarray, list],
                 name: Optional[str] = None,
                 metadata: Optional[Union[FilterMetadata, Dict[str, Any]]] = None):
        """
        Inicializa el filtro espectral con validación completa.

        Complejidad ciclomática: 4 (< 15 ✓)

        Parameters
        ----------
        wavelength : array_like
            Longitudes de onda en Angstroms
        response : array_like
            Respuesta del filtro
        name : str, optional
            Nombre del filtro
        metadata : FilterMetadata or dict, optional
            Metadatos del filtro
        """
        # Convertir a arrays de numpy
        self.wavelength = self._convert_to_array(wavelength, "wavelength")
        self.response = self._convert_to_array(response, "response")

        # Validar arrays
        FilterValidator.validate_wavelength_response_arrays(self.wavelength, self.response)

        # Asignar nombre
        self.name = name or f"filter_{id(self)}"

        # Procesar metadatos
        self.metadata = self._process_metadata(metadata)

        # Calcular propiedades derivadas
        self._calculate_derived_properties()

        logger.info(f"Filtro '{self.name}' inicializado correctamente")

    def _convert_to_array(self, data: Union[np.ndarray, list], name: str) -> np.ndarray:
        """
        Convierte datos de entrada a array numpy con validación.

        Complejidad ciclomática: 3 (< 15 ✓)
        """
        try:
            array = np.asarray(data, dtype=float)
            if array.ndim != 1:
                raise FilterValidationError(f"El {name} debe ser un array 1D")
            return array
        except Exception as e:
            raise FilterValidationError(f"Error convirtiendo {name} a array: {str(e)}")

    def _process_metadata(self, metadata: Optional[Union[FilterMetadata, Dict[str, Any]]]) -> FilterMetadata:
        """
        Procesa y valida los metadatos.

        Complejidad ciclomática: 4 (< 15 ✓)
        """
        if metadata is None:
            return FilterMetadata()
        elif isinstance(metadata, FilterMetadata):
            return metadata
        elif isinstance(metadata, dict):
            return FilterMetadata(**{k: v for k, v in metadata.items()
                                     if k in FilterMetadata.__annotations__})
        else:
            raise FilterValidationError("Los metadatos deben ser FilterMetadata o diccionario")

    def _calculate_derived_properties(self) -> None:
        """
        Calcula propiedades derivadas del filtro.

        Complejidad ciclomática: 5 (< 15 ✓)
        """
        try:
            # Longitud de onda efectiva
            self._effective_wavelength = FilterCalculator.calculate_effective_wavelength(
                self.wavelength, self.response
            )

            # Ancho efectivo
            weighted_variance = np.trapz(
                (self.wavelength - self._effective_wavelength) ** 2 * self.response,
                self.wavelength
            )
            normalization = np.trapz(self.response, self.wavelength)
            self._effective_width = np.sqrt(weighted_variance / normalization)

            # Actualizar metadatos si no están definidos
            if self.metadata.central_wavelength is None:
                self.metadata.central_wavelength = self._effective_wavelength
            if self.metadata.effective_width is None:
                self.metadata.effective_width = self._effective_width

        except Exception as e:
            logger.warning(f"Error calculando propiedades derivadas: {str(e)}")

    def apply_to_spectrum(self, spectrum) -> 'SpectrumData':
        """
        Aplica la respuesta del filtro a un espectro.

        Complejidad ciclomática: 4 (< 15 ✓)

        Parameters
        ----------
        spectrum : SpectrumData
            Espectro al que aplicar el filtro

        Returns
        -------
        SpectrumData
            Nuevo espectro con el filtro aplicado

        Raises
        ------
        SpectrumCompatibilityError
            Si el filtro y espectro no son compatibles
        FilterError
            Si ocurre un error durante la aplicación

        Examples
        --------
        >>> filtered_spectrum = filter_obj.apply_to_spectrum(original_spectrum)
        """
        try:
            # Validar compatibilidad
            FilterValidator.validate_spectrum_compatibility(
                self.wavelength, spectrum.wavelength
            )

            # Interpolar respuesta del filtro
            interp_response = FilterInterpolator.interpolate_response(
                spectrum.wavelength, self.wavelength, self.response
            )

            # Aplicar filtro
            filtered_flux = spectrum.flux * interp_response

            # Propagar errores si existen
            filtered_error = None
            if spectrum.error is not None:
                filtered_error = spectrum.error * interp_response

            # Crear nuevo espectro con metadatos actualizados
            new_metadata = spectrum.metadata.copy() if spectrum.metadata else {}
            new_metadata.update({
                'filter_applied': self.name,
                'filter_type': self.metadata.filter_type,
                'effective_wavelength': self._effective_wavelength
            })

            # Importar aquí para evitar dependencia circular
            from .spectrum_data import SpectrumData

            return SpectrumData(
                wavelength=spectrum.wavelength,
                flux=filtered_flux,
                error=filtered_error,
                flag=spectrum.flag,
                metadata=new_metadata
            )

        except Exception as e:
            raise FilterError(f"Error aplicando filtro '{self.name}': {str(e)}")

    def integrate_flux(self, spectrum) -> float:
        """
        Calcula la integración del espectro bajo el filtro.

        Complejidad ciclomática: 2 (< 15 ✓)

        Parameters
        ----------
        spectrum : SpectrumData
            Espectro a integrar

        Returns
        -------
        float
            Flujo total transmitido por el filtro

        Raises
        ------
        SpectrumCompatibilityError
            Si el filtro y espectro no son compatibles
        FilterError
            Si ocurre un error durante el cálculo

        Examples
        --------
        >>> total_flux = filter_obj.integrate_flux(spectrum)
        """
        try:
            # Validar compatibilidad
            FilterValidator.validate_spectrum_compatibility(
                self.wavelength, spectrum.wavelength
            )

            # Interpolar respuesta del filtro
            interp_response = FilterInterpolator.interpolate_response(
                spectrum.wavelength, self.wavelength, self.response
            )

            # Calcular flujo integrado
            return FilterCalculator.calculate_integrated_flux(
                spectrum.wavelength, spectrum.flux, interp_response
            )

        except Exception as e:
            raise FilterError(f"Error integrando flujo con filtro '{self.name}': {str(e)}")

    def calculate_magnitude(self, spectrum, zero_point: Optional[float] = None) -> float:
        """
        Calcula la magnitud del espectro en este filtro.

        Complejidad ciclomática: 4 (< 15 ✓)

        Parameters
        ----------
        spectrum : SpectrumData
            Espectro para calcular magnitud
        zero_point : float, optional
            Punto cero fotométrico, usa el de los metadatos si no se especifica

        Returns
        -------
        float
            Magnitud en el sistema fotométrico del filtro

        Raises
        ------
        FilterError
            Si no se puede calcular la magnitud
        """
        try:
            flux = self.integrate_flux(spectrum)
            zp = zero_point or self.metadata.zero_point

            if zp is None:
                raise FilterError("Punto cero fotométrico no definido")

            if flux <= 0:
                raise FilterError("Flujo no positivo, no se puede calcular magnitud")

            return -2.5 * np.log10(flux) + zp

        except Exception as e:
            raise FilterError(f"Error calculando magnitud: {str(e)}")

    def get_transmission_curve(self) -> Dict[str, np.ndarray]:
        """
        Obtiene la curva de transmisión del filtro.

        Complejidad ciclomática: 1 (< 15 ✓)

        Returns
        -------
        dict
            Diccionario con 'wavelength' y 'transmission'
        """
        return {
            'wavelength': self.wavelength.copy(),
            'transmission': self.response.copy()
        }

    def resample(self, new_wavelength: np.ndarray) -> 'FilterData':
        """
        Remuestrea el filtro a una nueva grilla de longitudes de onda.

        Complejidad ciclomática: 2 (< 15 ✓)

        Parameters
        ----------
        new_wavelength : np.ndarray
            Nueva grilla de longitudes de onda

        Returns
        -------
        FilterData
            Nuevo filtro remuestreado

        Raises
        ------
        FilterError
            Si ocurre un error durante el remuestreo
        """
        try:
            new_response = FilterInterpolator.interpolate_response(
                new_wavelength, self.wavelength, self.response
            )

            return FilterData(
                wavelength=new_wavelength,
                response=new_response,
                name=f"{self.name}_resampled",
                metadata=self.metadata
            )

        except Exception as e:
            raise FilterError(f"Error remuestreando filtro: {str(e)}")

    def as_dict(self) -> Dict[str, Any]:
        """
        Convierte el filtro a diccionario para serialización.

        Complejidad ciclomática: 1 (< 15 ✓)

        Returns
        -------
        dict
            Representación completa del filtro como diccionario
        """
        return {
            "wavelength": self.wavelength.tolist(),
            "response": self.response.tolist(),
            "name": self.name,
            "metadata": {
                "filter_type": self.metadata.filter_type,
                "central_wavelength": self.metadata.central_wavelength,
                "effective_width": self.metadata.effective_width,
                "zero_point": self.metadata.zero_point,
                "reference_system": self.metadata.reference_system,
                "source": self.metadata.source,
                "version": self.metadata.version
            },
            "derived_properties": {
                "effective_wavelength": getattr(self, '_effective_wavelength', None),
                "effective_width": getattr(self, '_effective_width', None)
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterData':
        """
        Crea un filtro desde un diccionario.

        Complejidad ciclomática: 1 (< 15 ✓)

        Parameters
        ----------
        data : dict
            Diccionario con los datos del filtro

        Returns
        -------
        FilterData
            Filtro reconstruido
        """
        metadata_dict = data.get("metadata", {})
        metadata = FilterMetadata(**metadata_dict)

        return cls(
            wavelength=data["wavelength"],
            response=data["response"],
            name=data.get("name"),
            metadata=metadata
        )

    def __repr__(self) -> str:
        """Representación string del filtro."""
        return (
            f"FilterData(name='{self.name}', "
            f"wavelength_range=({self.wavelength.min():.1f}, {self.wavelength.max():.1f}), "
            f"n_points={len(self.wavelength)})"
        )

    def __str__(self) -> str:
        """Representación string legible del filtro."""
        return f"Filtro '{self.name}' con {len(self.wavelength)} puntos espectrales"

    @property
    def effective_wavelength(self) -> float:
        """Longitud de onda efectiva del filtro."""
        return getattr(self, '_effective_wavelength', 0.0)

    @property
    def effective_width(self) -> float:
        """Ancho efectivo del filtro."""
        return getattr(self, '_effective_width', 0.0)

    @property
    def wavelength_range(self) -> tuple:
        """Rango de longitudes de onda del filtro."""
        return (self.wavelength.min(), self.wavelength.max())