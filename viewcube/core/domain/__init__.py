"""Modelos y entidades del dominio."""
from .entities import *
from .models import *

__all__ = [
    "AstronomicalObject", "SpatialCoordinate", "WavelengthRange",
    "Spaxel", "Survey", "InstrumentType", "FiberType",
    "CubeData", "FilterData", "SpectrumData"
]