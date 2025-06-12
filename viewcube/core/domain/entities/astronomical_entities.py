"""
Entidades astronómicas del dominio.

Este módulo contiene las entidades que representan conceptos astronómicos
fundamentales utilizados en ViewCube.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum


class InstrumentType(Enum):
    """Tipos de instrumentos astronómicos soportados."""
    CALIFA = "CALIFA"
    MANGA = "MaNGA"
    MUSE = "MUSE"
    WEAVE = "WEAVE"
    MEGARA = "MEGARA"
    PPAK = "PPAK"
    UNKNOWN = "UNKNOWN"


class FiberType(Enum):
    """Tipos de fibra óptica."""
    CIRCULAR = "C"
    HEXAGONAL = "H"


class SurveyType(Enum):
    """Tipos de surveys astronómicos."""
    CALIFA = "CALIFA"
    MANGA = "MaNGA"
    MUSE = "MUSE"
    WEAVE = "WEAVE"
    OTHER = "OTHER"


@dataclass
class SpatialCoordinate:
    """Representa una coordenada espacial en el cielo."""

    def __init__(self, x: float, y: float, ra: Optional[float] = None,
                 dec: Optional[float] = None, unit: str = "arcsec"):
        """
        Inicializa una coordenada espacial.

        Args:
            x: Coordenada X en el plano del detector
            y: Coordenada Y en el plano del detector
            ra: Ascensión recta en grados (opcional)
            dec: Declinación en grados (opcional)
            unit: Unidad de las coordenadas
        """
        self.x = float(x)
        self.y = float(y)
        self.ra = float(ra) if ra is not None else None
        self.dec = float(dec) if dec is not None else None
        self.unit = unit

    def distance_to(self, other: 'SpatialCoordinate') -> float:
        """Calcula la distancia a otra coordenada."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def offset_from(self, reference: 'SpatialCoordinate') -> Tuple[float, float]:
        """Calcula el offset desde una coordenada de referencia."""
        return (self.x - reference.x, self.y - reference.y)

    def as_dict(self) -> Dict[str, Any]:
        """Convierte la coordenada a diccionario."""
        return {
            "x": self.x,
            "y": self.y,
            "ra": self.ra,
            "dec": self.dec,
            "unit": self.unit
        }


@dataclass
class WavelengthRange:
    """Representa un rango de longitudes de onda."""

    def __init__(self, min_wavelength: float, max_wavelength: float,
                 unit: str = "Angstrom", resolution: Optional[float] = None):
        """
        Inicializa un rango de longitudes de onda.

        Args:
            min_wavelength: Longitud de onda mínima
            max_wavelength: Longitud de onda máxima
            unit: Unidad de longitud de onda
            resolution: Resolución espectral (opcional)
        """
        self.min_wavelength = float(min_wavelength)
        self.max_wavelength = float(max_wavelength)
        self.unit = unit
        self.resolution = float(resolution) if resolution is not None else None

    def contains(self, wavelength: float) -> bool:
        """Verifica si una longitud de onda está en el rango."""
        return self.min_wavelength <= wavelength <= self.max_wavelength

    def width(self) -> float:
        """Retorna el ancho del rango."""
        return self.max_wavelength - self.min_wavelength

    def center(self) -> float:
        """Retorna la longitud de onda central."""
        return (self.min_wavelength + self.max_wavelength) / 2.0

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el rango a diccionario."""
        return {
            "min_wavelength": self.min_wavelength,
            "max_wavelength": self.max_wavelength,
            "unit": self.unit,
            "resolution": self.resolution
        }


class Spaxel:
    """Representa un elemento espacial (spaxel) en un cubo de datos."""

    def __init__(self, id: int, coordinate: SpatialCoordinate,
                 radius: Optional[float] = None, fiber_type: FiberType = FiberType.CIRCULAR):
        """
        Inicializa un spaxel.

        Args:
            id: Identificador único del spaxel
            coordinate: Coordenada espacial del spaxel
            radius: Radio del spaxel
            fiber_type: Tipo de fibra (circular o hexagonal)
        """
        self.id = int(id)
        self.coordinate = coordinate
        self.radius = float(radius) if radius is not None else None
        self.fiber_type = fiber_type
        self._spectrum_data = None

    def set_spectrum_data(self, spectrum_data):
        """Asocia datos espectrales al spaxel."""
        self._spectrum_data = spectrum_data

    def get_spectrum_data(self):
        """Obtiene los datos espectrales del spaxel."""
        return self._spectrum_data

    def area(self) -> Optional[float]:
        """Calcula el área del spaxel."""
        if self.radius is None:
            return None

        if self.fiber_type == FiberType.CIRCULAR:
            return np.pi * self.radius ** 2
        elif self.fiber_type == FiberType.HEXAGONAL:
            # Área de hexágono regular
            return 3 * np.sqrt(3) / 2 * self.radius ** 2

        return None

    def distance_to(self, other: 'Spaxel') -> float:
        """Calcula la distancia a otro spaxel."""
        return self.coordinate.distance_to(other.coordinate)

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el spaxel a diccionario."""
        return {
            "id": self.id,
            "coordinate": self.coordinate.as_dict(),
            "radius": self.radius,
            "fiber_type": self.fiber_type.value,
            "area": self.area()
        }


class Instrument:
    """Representa un instrumento astronómico."""

    def __init__(self, name: str, instrument_type: InstrumentType,
                 wavelength_range: Optional[WavelengthRange] = None,
                 spatial_resolution: Optional[float] = None,
                 spectral_resolution: Optional[float] = None):
        """
        Inicializa un instrumento.

        Args:
            name: Nombre del instrumento
            instrument_type: Tipo de instrumento
            wavelength_range: Rango de longitudes de onda
            spatial_resolution: Resolución espacial
            spectral_resolution: Resolución espectral
        """
        self.name = str(name)
        self.instrument_type = instrument_type
        self.wavelength_range = wavelength_range
        self.spatial_resolution = float(spatial_resolution) if spatial_resolution is not None else None
        self.spectral_resolution = float(spectral_resolution) if spectral_resolution is not None else None

    def supports_wavelength(self, wavelength: float) -> bool:
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


class Observatory:
    """Representa un observatorio astronómico."""

    def __init__(self, name: str, location: Optional[str] = None,
                 altitude: Optional[float] = None, instruments: Optional[list] = None):
        """
        Inicializa un observatorio.

        Args:
            name: Nombre del observatorio
            location: Ubicación geográfica
            altitude: Altitud en metros
            instruments: Lista de instrumentos disponibles
        """
        self.name = str(name)
        self.location = location
        self.altitude = float(altitude) if altitude is not None else None
        self.instruments = instruments if instruments is not None else []

    def add_instrument(self, instrument: Instrument) -> None:
        """Añade un instrumento al observatorio."""
        self.instruments.append(instrument)

    def get_instrument(self, name: str) -> Optional[Instrument]:
        """Obtiene un instrumento por nombre."""
        for instrument in self.instruments:
            if instrument.name == name:
                return instrument
        return None

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el observatorio a diccionario."""
        return {
            "name": self.name,
            "location": self.location,
            "altitude": self.altitude,
            "instruments": [inst.as_dict() for inst in self.instruments]
        }


class AstronomicalObject:
    """Representa un objeto astronómico observado."""

    def __init__(self, name: str, object_type: str = "GALAXY",
                 redshift: Optional[float] = None, velocity: Optional[float] = None,
                 coordinates: Optional[SpatialCoordinate] = None,
                 distance_pc: Optional[float] = None):
        """
        Inicializa un objeto astronómico.

        Args:
            name: Nombre del objeto
            object_type: Tipo de objeto (GALAXY, STAR, etc.)
            redshift: Corrimiento al rojo
            velocity: Velocidad radial en km/s
            coordinates: Coordenadas del objeto
            distance_pc: Distancia en parsecs
        """
        self.name = str(name)
        self.object_type = str(object_type)
        self.redshift = float(redshift) if redshift is not None else None
        self.velocity = float(velocity) if velocity is not None else None
        self.coordinates = coordinates
        self.distance_pc = float(distance_pc) if distance_pc is not None else None
        self._cube_data = None
        self._metadata = {}

    def set_cube_data(self, cube_data):
        """Asocia datos de cubo al objeto."""
        self._cube_data = cube_data

    def get_cube_data(self):
        """Obtiene los datos de cubo del objeto."""
        return self._cube_data

    def set_metadata(self, key: str, value: Any) -> None:
        """Establece un metadato del objeto."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Obtiene un metadato del objeto."""
        return self._metadata.get(key, default)

    def calculate_redshift_from_velocity(self, c: float = 299792.458) -> Optional[float]:
        """Calcula el redshift a partir de la velocidad."""
        if self.velocity is not None:
            return self.velocity / c
        return None

    def calculate_velocity_from_redshift(self, c: float = 299792.458) -> Optional[float]:
        """Calcula la velocidad a partir del redshift."""
        if self.redshift is not None:
            return self.redshift * c
        return None

    def rest_wavelength(self, observed_wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convierte longitud de onda observada a longitud de onda en reposo."""
        if self.redshift is not None:
            return observed_wavelength / (1.0 + self.redshift)
        return observed_wavelength

    def observed_wavelength(self, rest_wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convierte longitud de onda en reposo a longitud de onda observada."""
        if self.redshift is not None:
            return rest_wavelength * (1.0 + self.redshift)
        return rest_wavelength

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el objeto a diccionario."""
        return {
            "name": self.name,
            "object_type": self.object_type,
            "redshift": self.redshift,
            "velocity": self.velocity,
            "coordinates": self.coordinates.as_dict() if self.coordinates else None,
            "distance_pc": self.distance_pc,
            "metadata": self._metadata
        }


class Survey:
    """Representa un survey astronómico."""

    def __init__(self, name: str, survey_type: SurveyType,
                 observatory: Optional[Observatory] = None,
                 instrument: Optional[Instrument] = None,
                 description: Optional[str] = None):
        """
        Inicializa un survey.

        Args:
            name: Nombre del survey
            survey_type: Tipo de survey
            observatory: Observatorio asociado
            instrument: Instrumento principal
            description: Descripción del survey
        """
        self.name = str(name)
        self.survey_type = survey_type
        self.observatory = observatory
        self.instrument = instrument
        self.description = description
        self._objects = []

    def add_object(self, astronomical_object: AstronomicalObject) -> None:
        """Añade un objeto al survey."""
        self._objects.append(astronomical_object)

    def get_objects(self) -> list:
        """Obtiene todos los objetos del survey."""
        return self._objects.copy()

    def get_object_by_name(self, name: str) -> Optional[AstronomicalObject]:
        """Obtiene un objeto por nombre."""
        for obj in self._objects:
            if obj.name == name:
                return obj
        return None

    def as_dict(self) -> Dict[str, Any]:
        """Convierte el survey a diccionario."""
        return {
            "name": self.name,
            "survey_type": self.survey_type.value,
            "observatory": self.observatory.as_dict() if self.observatory else None,
            "instrument": self.instrument.as_dict() if self.instrument else None,
            "description": self.description,
            "objects_count": len(self._objects)
        }
