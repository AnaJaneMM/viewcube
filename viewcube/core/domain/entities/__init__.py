"""Entidades del dominio astronómico."""
from .astronomical_entities import (
    AstronomicalObject,
    SpatialCoordinate,
    WavelengthRange,
    Spaxel,
    Survey,
    InstrumentType,
    FiberType
)

__all__ = [
    "AstronomicalObject", "SpatialCoordinate", "WavelengthRange",
    "Spaxel", "Survey", "InstrumentType", "FiberType"
]