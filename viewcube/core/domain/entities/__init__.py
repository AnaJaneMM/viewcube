"""
Entidades del dominio astronómico.

Este módulo exporta las entidades fundamentales del dominio astronómico
utilizadas en ViewCube, incluyendo objetos astronómicos, instrumentos,
observatorios y coordenadas espaciales.

Versión: 2.0.0
"""

from .astronomical_entities import (
    # Enumeraciones
    InstrumentType,
    FiberType,
    SurveyType,

    # Interfaces base
    SpatialEntity,
    SpectralEntity,

    # Entidades principales
    SpatialCoordinate,
    WavelengthRange,
    Spaxel,
    AstronomicalObject,
    Instrument,
    Observatory,
    Survey,

    # Clases utilitarias
    SpectralCalculator,
    RedshiftCalculator,

    # Factory y validadores
    AstronomicalEntityFactory,
    DataValidator
)

# Versión del módulo de entidades
__version__ = "2.0.0"

# Exportaciones públicas principales
__all__ = [
    # Enumeraciones
    "InstrumentType",
    "FiberType",
    "SurveyType",

    # Interfaces
    "SpatialEntity",
    "SpectralEntity",

    # Entidades de dominio
    "SpatialCoordinate",
    "WavelengthRange",
    "Spaxel",
    "AstronomicalObject",
    "Instrument",
    "Observatory",
    "Survey",

    # Utilidades
    "SpectralCalculator",
    "RedshiftCalculator",
    "AstronomicalEntityFactory",
    "DataValidator"
]

# Metadatos del módulo
__email__ = "viewcube@astronomy.org"
__status__ = "Production"

# Configuración de logging para entidades
import logging
logger = logging.getLogger(__name__)
logger.info("Módulo de entidades astronómicas cargado correctamente")