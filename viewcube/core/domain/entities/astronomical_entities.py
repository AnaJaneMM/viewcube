"""
Entidades astronómicas del dominio.

Este módulo contiene las entidades que representan conceptos astronómicos
fundamentales utilizados en ViewCube. Refactorización que mantiene toda la
funcionalidad original con mejoras en modularidad y reducción de complejidad.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class InstrumentType(Enum):
    """Tipos de instrumentos astronómicos soportados."""
    CALIFA = "CALIFA"
    MANGA = "MaNGA"
    MUSE = "MUSE"
    WEAVE = "WEAVE"
    MEGARA = "MEGARA"
    PPAK = "PPAK"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_header(cls, header: Dict[str, Any]) -> 'InstrumentType':
        """Detecta el tipo de instrumento desde un header FITS."""
        instrument = header.get('INSTRUME', '').strip().upper()
        survey = header.get('SURVEY', '').strip().upper()

        # Mapeo de instrumentos conocidos
        instrument_mapping = {
            'CALIFA': cls.CALIFA,
            'MANGA': cls.MANGA,
            'MUSE': cls.MUSE,
            'WEAVE': cls.WEAVE,
            'MEGARA': cls.MEGARA,
            'PPAK': cls.PPAK
        }

        # Primero intentar por instrumento
        for key, value in instrument_mapping.items():
            if key in instrument:
                return value

        # Luego intentar por survey
        for key, value in instrument_mapping.items():
            if key in survey:
                return value

        return cls.UNKNOWN


class FiberType(Enum):
    """Tipos de fibra óptica."""
    CIRCULAR = "C"
    HEXAGONAL = "H"

    @classmethod
    def from_shape_code(cls, code: str) -> 'FiberType':
        """Convierte código de forma a tipo de fibra."""
        code = code.upper().strip()
        if code == "C":
            return cls.CIRCULAR
        elif code == "H":
            return cls.HEXAGONAL
        else:
            return cls.CIRCULAR  # Default


class SurveyType(Enum):
    """Tipos de surveys astronómicos."""
    CALIFA = "CALIFA"
    MANGA = "MaNGA"
    MUSE = "MUSE"
    WEAVE = "WEAVE"
    OTHER = "OTHER"


# Interfaces y clases base para reducir acoplamiento
class SpatialEntity(ABC):
    """Interfaz para entidades con información espacial."""

    @abstractmethod
    def get_position(self) -> Tuple[float, float]:
        """Retorna la posición espacial como (x, y)."""
        pass

    @abstractmethod
    def distance_to(self, other: 'SpatialEntity') -> float:
        """Calcula distancia a otra entidad espacial."""
        pass


class SpectralEntity(ABC):
    """Interfaz para entidades con información espectral."""

    @abstractmethod
    def get_wavelength_range(self) -> Tuple[float, float]:
        """Retorna el rango de longitudes de onda como (min, max)."""
        pass


@dataclass(frozen=True)
class SpatialCoordinate(SpatialEntity):
    """Representa una coordenada espacial en el cielo."""
    x: float
    y: float
    ra: Optional[float] = None
    dec: Optional[float] = None
    unit: str = "arcsec"

    def __post_init__(self):
        """Validación de coordenadas."""
        if not isinstance(self.x, (int, float)) or not isinstance(self.y, (int, float)):
            raise ValueError("Las coordenadas x e y deben ser números")

        if self.ra is not None and (self.ra < 0 or self.ra >= 360):
            raise ValueError("RA debe estar entre 0 y 360 grados")

        if self.dec is not None and (self.dec < -90 or self.dec > 90):
            raise ValueError("Dec debe estar entre -90 y 90 grados")

    def get_position(self) -> Tuple[float, float]:
        """Retorna la posición como tupla (x, y)."""
        return (self.x, self.y)

    def distance_to(self, other: Union['SpatialCoordinate', SpatialEntity]) -> float:
        """Calcula la distancia euclidiana a otra coordenada."""
        if isinstance(other, SpatialCoordinate):
            other_x, other_y = other.x, other.y
        else:
            other_x, other_y = other.get_position()

        return np.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)

    def offset_from(self, reference: 'SpatialCoordinate') -> Tuple[float, float]:
        """Calcula el offset desde una coordenada de referencia."""
        return (self.x - reference.x, self.y - reference.y)

    def to_skycoord(self) -> Tuple[Optional[float], Optional[float]]:
        """Convierte a coordenadas celestiales (RA, Dec)."""
        return (self.ra, self.dec)

    def as_dict(self) -> Dict[str, Any]:
        """Convierte la coordenada a diccionario."""
        return {
            "x": self.x,
            "y": self.y,
            "ra": self.ra,
            "dec": self.dec,
            "unit": self.unit
        }


@dataclass(frozen=True)
class WavelengthRange(SpectralEntity):
    """Representa un rango de longitudes de onda con validación."""
    min_wavelength: float
    max_wavelength: float
    unit: str = "Angstrom"
    resolution: Optional[float] = None

    def __post_init__(self):
        """Validación del rango de longitudes de onda."""
        if self.min_wavelength >= self.max_wavelength:
            raise ValueError("La longitud de onda mínima debe ser menor que la máxima")

        if self.min_wavelength <= 0 or self.max_wavelength <= 0:
            raise ValueError("Las longitudes de onda deben ser positivas")

        if self.resolution is not None and self.resolution <= 0:
            raise ValueError("La resolución debe ser positiva")

    def get_wavelength_range(self) -> Tuple[float, float]:
        """Retorna el rango como tupla (min, max)."""
        return (self.min_wavelength, self.max_wavelength)

    def contains(self, wavelength: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Verifica si una longitud de onda está en el rango."""
        if isinstance(wavelength, np.ndarray):
            return (wavelength >= self.min_wavelength) & (wavelength <= self.max_wavelength)
        return self.min_wavelength <= wavelength <= self.max_wavelength

    def width(self) -> float:
        """Retorna el ancho del rango."""
        return self.max_wavelength - self.min_wavelength

    def center(self) -> float:
        """Retorna la longitud de onda central."""
        return (self.min_wavelength + self.max_wavelength) / 2.0

    def overlap_with(self, other: 'WavelengthRange') -> Optional['WavelengthRange']:
        """Calcula el solapamiento con otro rango."""
        overlap_min = max(self.min_wavelength, other.min_wavelength)
        overlap_max = min(self.max_wavelength, other.max_wavelength)

        if overlap_min < overlap_max:
            return WavelengthRange(overlap_min, overlap_max, self.unit)
        return None

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el rango a diccionario."""
        return {
            "min_wavelength": self.min_wavelength,
            "max_wavelength": self.max_wavelength,
            "unit": self.unit,
            "resolution": self.resolution,
            "width": self.width(),
            "center": self.center()
        }


# Clases auxiliares para cálculos específicos
class SpectralCalculator:
    """Utilidades para cálculos espectrales."""

    @staticmethod
    def calculate_area_circular(radius: float) -> float:
        """Calcula área de fibra circular."""
        return np.pi * radius ** 2

    @staticmethod
    def calculate_area_hexagonal(radius: float) -> float:
        """Calcula área de fibra hexagonal."""
        return 3 * np.sqrt(3) / 2 * radius ** 2

    @staticmethod
    def signal_to_noise_ratio(data: np.ndarray, error: Optional[np.ndarray] = None) -> float:
        """Calcula relación señal-ruido."""
        if error is None or len(error) == 0:
            return 0.0

        valid_mask = np.isfinite(data) & np.isfinite(error) & (error > 0)
        if not np.any(valid_mask):
            return 0.0

        valid_data = data[valid_mask]
        valid_error = error[valid_mask]

        if len(valid_data) == 0:
            return 0.0

        return np.mean(valid_data) / np.mean(valid_error)


@dataclass
class Spaxel(SpatialEntity, SpectralEntity):
    """Representa un elemento espacial (spaxel) en un cubo de datos."""
    id: int
    coordinate: SpatialCoordinate
    radius: Optional[float] = None
    fiber_type: FiberType = FiberType.CIRCULAR
    _spectrum_data: Optional[np.ndarray] = field(default=None, init=False)
    _wavelength_range: Optional[WavelengthRange] = field(default=None, init=False)
    _error_data: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        """Validación de datos del spaxel."""
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("El ID debe ser un entero no negativo")

        if self.radius is not None and self.radius <= 0:
            raise ValueError("El radio debe ser positivo")

    def get_position(self) -> Tuple[float, float]:
        """Retorna posición del spaxel."""
        return self.coordinate.get_position()

    def distance_to(self, other: Union['Spaxel', SpatialEntity]) -> float:
        """Calcula distancia a otro spaxel."""
        return self.coordinate.distance_to(other)

    def get_wavelength_range(self) -> Tuple[float, float]:
        """Retorna rango de longitudes de onda."""
        if self._wavelength_range is None:
            return (0.0, 0.0)
        return self._wavelength_range.get_wavelength_range()

    def set_spectrum_data(self, spectrum_data: np.ndarray,
                          wavelength_range: Optional[WavelengthRange] = None,
                          error_data: Optional[np.ndarray] = None) -> None:
        """Asocia datos espectrales al spaxel."""
        if not isinstance(spectrum_data, np.ndarray):
            raise ValueError("Los datos espectrales deben ser un array numpy")

        self._spectrum_data = spectrum_data.copy()

        if wavelength_range is not None:
            self._wavelength_range = wavelength_range

        if error_data is not None:
            if error_data.shape != spectrum_data.shape:
                raise ValueError("Los datos de error deben tener la misma forma que los datos espectrales")
            self._error_data = error_data.copy()

    def get_spectrum_data(self) -> Optional[np.ndarray]:
        """Obtiene los datos espectrales del spaxel."""
        return self._spectrum_data.copy() if self._spectrum_data is not None else None

    def get_error_data(self) -> Optional[np.ndarray]:
        """Obtiene los datos de error del spaxel."""
        return self._error_data.copy() if self._error_data is not None else None

    def area(self) -> Optional[float]:
        """Calcula el área del spaxel."""
        if self.radius is None:
            return None

        if self.fiber_type == FiberType.CIRCULAR:
            return SpectralCalculator.calculate_area_circular(self.radius)
        elif self.fiber_type == FiberType.HEXAGONAL:
            return SpectralCalculator.calculate_area_hexagonal(self.radius)

        return None

    def signal_to_noise_ratio(self) -> float:
        """Calcula la relación señal-ruido del spaxel."""
        if self._spectrum_data is None:
            return 0.0

        return SpectralCalculator.signal_to_noise_ratio(self._spectrum_data, self._error_data)

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el spaxel a diccionario."""
        return {
            "id": self.id,
            "coordinate": self.coordinate.as_dict(),
            "radius": self.radius,
            "fiber_type": self.fiber_type.value,
            "area": self.area(),
            "wavelength_range": self._wavelength_range.as_dict() if self._wavelength_range else None,
            "signal_to_noise": self.signal_to_noise_ratio(),
            "has_spectrum": self._spectrum_data is not None,
            "has_error": self._error_data is not None
        }


# Clases auxiliares para cálculos astrofísicos
class RedshiftCalculator:
    """Utilidades para cálculos de corrimiento al rojo."""

    @staticmethod
    def velocity_to_redshift(velocity: float, c: float = 299792.458) -> float:
        """Convierte velocidad a redshift."""
        return velocity / c

    @staticmethod
    def redshift_to_velocity(redshift: float, c: float = 299792.458) -> float:
        """Convierte redshift a velocidad."""
        return redshift * c

    @staticmethod
    def observed_to_rest_wavelength(observed_wavelength: Union[float, np.ndarray],
                                    redshift: float) -> Union[float, np.ndarray]:
        """Convierte longitud de onda observada a reposo."""
        return observed_wavelength / (1.0 + redshift)

    @staticmethod
    def rest_to_observed_wavelength(rest_wavelength: Union[float, np.ndarray],
                                    redshift: float) -> Union[float, np.ndarray]:
        """Convierte longitud de onda en reposo a observada."""
        return rest_wavelength * (1.0 + redshift)


@dataclass
class AstronomicalObject:
    """Representa un objeto astronómico observado."""
    name: str
    object_type: str = "GALAXY"
    redshift: Optional[float] = None
    velocity: Optional[float] = None
    coordinates: Optional[SpatialCoordinate] = None
    distance_pc: Optional[float] = None
    _cube_data: Optional[Any] = field(default=None, init=False)
    _metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Validación y sincronización de parámetros."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("El nombre del objeto debe ser una cadena no vacía")

        if self.redshift is not None and self.redshift < 0:
            raise ValueError("El redshift no puede ser negativo")

        if self.distance_pc is not None and self.distance_pc <= 0:
            raise ValueError("La distancia debe ser positiva")

        # Sincronizar redshift y velocidad si solo uno está presente
        if self.redshift is not None and self.velocity is None:
            self.velocity = RedshiftCalculator.redshift_to_velocity(self.redshift)
        elif self.velocity is not None and self.redshift is None:
            self.redshift = RedshiftCalculator.velocity_to_redshift(self.velocity)

    def set_cube_data(self, cube_data: Any) -> None:
        """Asocia datos de cubo al objeto."""
        self._cube_data = cube_data

    def get_cube_data(self) -> Any:
        """Obtiene los datos de cubo del objeto."""
        return self._cube_data

    def set_metadata(self, key: str, value: Any) -> None:
        """Establece un metadato del objeto."""
        if not isinstance(key, str) or not key:
            raise ValueError("La clave de metadato debe ser una cadena no vacía")
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Obtiene un metadato del objeto."""
        return self._metadata.get(key, default)

    def get_all_metadata(self) -> Dict[str, Any]:
        """Obtiene todos los metadatos."""
        return self._metadata.copy()

    def calculate_redshift_from_velocity(self, c: float = 299792.458) -> Optional[float]:
        """Calcula el redshift a partir de la velocidad."""
        if self.velocity is not None:
            calculated_z = RedshiftCalculator.velocity_to_redshift(self.velocity, c)
            self.redshift = calculated_z
            return calculated_z
        return None

    def calculate_velocity_from_redshift(self, c: float = 299792.458) -> Optional[float]:
        """Calcula la velocidad a partir del redshift."""
        if self.redshift is not None:
            calculated_v = RedshiftCalculator.redshift_to_velocity(self.redshift, c)
            self.velocity = calculated_v
            return calculated_v
        return None

    def rest_wavelength(self, observed_wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convierte longitud de onda observada a longitud de onda en reposo."""
        if self.redshift is not None:
            return RedshiftCalculator.observed_to_rest_wavelength(observed_wavelength, self.redshift)
        return observed_wavelength

    def observed_wavelength(self, rest_wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convierte longitud de onda en reposo a longitud de onda observada."""
        if self.redshift is not None:
            return RedshiftCalculator.rest_to_observed_wavelength(rest_wavelength, self.redshift)
        return rest_wavelength

    def angular_distance(self, other: 'AstronomicalObject') -> Optional[float]:
        """Calcula distancia angular a otro objeto."""
        if self.coordinates is None or other.coordinates is None:
            return None
        return self.coordinates.distance_to(other.coordinates)

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el objeto a diccionario."""
        return {
            "name": self.name,
            "object_type": self.object_type,
            "redshift": self.redshift,
            "velocity": self.velocity,
            "coordinates": self.coordinates.as_dict() if self.coordinates else None,
            "distance_pc": self.distance_pc,
            "metadata": self._metadata.copy(),
            "has_cube_data": self._cube_data is not None
        }


@dataclass
class Instrument:
    """Representa un instrumento astronómico."""
    name: str
    instrument_type: InstrumentType
    wavelength_range: Optional[WavelengthRange] = None
    spatial_resolution: Optional[float] = None
    spectral_resolution: Optional[float] = None

    def __post_init__(self):
        """Validación de parámetros del instrumento."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("El nombre del instrumento debe ser una cadena no vacía")

        if self.spatial_resolution is not None and self.spatial_resolution <= 0:
            raise ValueError("La resolución espacial debe ser positiva")

        if self.spectral_resolution is not None and self.spectral_resolution <= 0:
            raise ValueError("La resolución espectral debe ser positiva")

    def supports_wavelength(self, wavelength: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Verifica si el instrumento soporta una longitud de onda."""
        if self.wavelength_range is None:
            return True
        return self.wavelength_range.contains(wavelength)

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el instrumento a diccionario."""
        return {
            "name": self.name,
            "instrument_type": self.instrument_type.value,
            "wavelength_range": self.wavelength_range.as_dict() if self.wavelength_range else None,
            "spatial_resolution": self.spatial_resolution,
            "spectral_resolution": self.spectral_resolution
        }


@dataclass
class Observatory:
    """Representa un observatorio astronómico."""
    name: str
    location: Optional[str] = None
    altitude: Optional[float] = None
    instruments: List[Instrument] = field(default_factory=list)

    def __post_init__(self):
        """Validación de parámetros del observatorio."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("El nombre del observatorio debe ser una cadena no vacía")

        if self.altitude is not None and self.altitude < 0:
            raise ValueError("La altitud no puede ser negativa")

    def add_instrument(self, instrument: Instrument) -> None:
        """Añade un instrumento al observatorio."""
        if not isinstance(instrument, Instrument):
            raise ValueError("Debe proporcionar una instancia válida de Instrument")

        # Verificar que no existe ya un instrumento con el mismo nombre
        existing_names = [inst.name for inst in self.instruments]
        if instrument.name in existing_names:
            raise ValueError(f"Ya existe un instrumento con el nombre '{instrument.name}'")

        self.instruments.append(instrument)

    def get_instrument(self, name: str) -> Optional[Instrument]:
        """Obtiene un instrumento por nombre."""
        for instrument in self.instruments:
            if instrument.name == name:
                return instrument
        return None

    def get_instruments_by_type(self, instrument_type: InstrumentType) -> List[Instrument]:
        """Obtiene instrumentos por tipo."""
        return [inst for inst in self.instruments if inst.instrument_type == instrument_type]

    def remove_instrument(self, name: str) -> bool:
        """Elimina un instrumento por nombre."""
        for i, instrument in enumerate(self.instruments):
            if instrument.name == name:
                del self.instruments[i]
                return True
        return False

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el observatorio a diccionario."""
        return {
            "name": self.name,
            "location": self.location,
            "altitude": self.altitude,
            "instruments": [inst.as_dict() for inst in self.instruments],
            "instrument_count": len(self.instruments)
        }


@dataclass
class Survey:
    """Representa un survey astronómico."""
    name: str
    survey_type: SurveyType
    observatory: Optional[Observatory] = None
    instrument: Optional[Instrument] = None
    description: Optional[str] = None
    _objects: List[AstronomicalObject] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validación de parámetros del survey."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("El nombre del survey debe ser una cadena no vacía")

    def add_object(self, astronomical_object: AstronomicalObject) -> None:
        """Añade un objeto al survey."""
        if not isinstance(astronomical_object, AstronomicalObject):
            raise ValueError("Debe proporcionar una instancia válida de AstronomicalObject")

        # Verificar que no existe ya un objeto con el mismo nombre
        existing_names = [obj.name for obj in self._objects]
        if astronomical_object.name in existing_names:
            raise ValueError(f"Ya existe un objeto con el nombre '{astronomical_object.name}'")

        self._objects.append(astronomical_object)

    def get_objects(self) -> List[AstronomicalObject]:
        """Obtiene todos los objetos del survey."""
        return self._objects.copy()

    def get_object_by_name(self, name: str) -> Optional[AstronomicalObject]:
        """Obtiene un objeto por nombre."""
        for obj in self._objects:
            if obj.name == name:
                return obj
        return None

    def get_objects_by_type(self, object_type: str) -> List[AstronomicalObject]:
        """Obtiene objetos por tipo."""
        return [obj for obj in self._objects if obj.object_type == object_type]

    def remove_object(self, name: str) -> bool:
        """Elimina un objeto por nombre."""
        for i, obj in enumerate(self._objects):
            if obj.name == name:
                del self._objects[i]
                return True
        return False

    def get_redshift_range(self) -> Tuple[Optional[float], Optional[float]]:
        """Obtiene el rango de redshifts de los objetos."""
        redshifts = [obj.redshift for obj in self._objects if obj.redshift is not None]
        if not redshifts:
            return (None, None)
        return (min(redshifts), max(redshifts))

    def get_object_count(self) -> int:
        """Obtiene el número de objetos en el survey."""
        return len(self._objects)

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el survey a diccionario."""
        return {
            "name": self.name,
            "survey_type": self.survey_type.value,
            "observatory": self.observatory.as_dict() if self.observatory else None,
            "instrument": self.instrument.as_dict() if self.instrument else None,
            "description": self.description,
            "objects_count": len(self._objects),
            "redshift_range": self.get_redshift_range()
        }


# Factory para crear entidades desde datos externos
class AstronomicalEntityFactory:
    """Factory para crear entidades astronómicas desde diversos formatos de datos."""

    @staticmethod
    def create_spatial_coordinate(x: float, y: float, **kwargs) -> SpatialCoordinate:
        """Crea una coordenada espacial con validación."""
        return SpatialCoordinate(x=x, y=y, **kwargs)

    @staticmethod
    def create_wavelength_range(min_wave: float, max_wave: float, **kwargs) -> WavelengthRange:
        """Crea un rango de longitudes de onda con validación."""
        return WavelengthRange(min_wavelength=min_wave, max_wavelength=max_wave, **kwargs)

    @staticmethod
    def create_spaxel_from_data(spaxel_id: int, x: float, y: float,
                                spectrum: Optional[np.ndarray] = None,
                                **kwargs) -> Spaxel:
        """Crea un spaxel desde datos básicos."""
        coordinate = SpatialCoordinate(x=x, y=y)
        spaxel = Spaxel(id=spaxel_id, coordinate=coordinate, **kwargs)

        if spectrum is not None:
            spaxel.set_spectrum_data(spectrum)

        return spaxel

    @staticmethod
    def create_instrument_from_header(header: Dict[str, Any]) -> Instrument:
        """Crea un instrumento desde un header FITS."""
        instrument_type = InstrumentType.from_header(header)
        instrument_name = header.get('INSTRUME', 'Unknown').strip()

        return Instrument(
            name=instrument_name,
            instrument_type=instrument_type
        )


# Validadores para mantener integridad de datos
class DataValidator:
    """Validadores para entidades astronómicas."""

    @staticmethod
    def validate_wavelength_data(wavelengths: np.ndarray) -> bool:
        """Valida un array de longitudes de onda."""
        if not isinstance(wavelengths, np.ndarray):
            return False

        if len(wavelengths) == 0:
            return False

        if not np.all(np.isfinite(wavelengths)):
            return False

        if not np.all(wavelengths > 0):
            return False

        # Verificar que está ordenado
        if not np.all(wavelengths[1:] >= wavelengths[:-1]):
            return False

        return True

    @staticmethod
    def validate_spectrum_data(spectrum: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> bool:
        """Valida datos espectrales."""
        if not isinstance(spectrum, np.ndarray):
            return False

        if len(spectrum) == 0:
            return False

        if wavelengths is not None and len(spectrum) != len(wavelengths):
            return False

        return True