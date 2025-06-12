"""Modelos de datos del dominio."""

from .cube_data import CubeData
from .filter_data import FilterData
from .spectrum_data import SpectrumData

__all__ = [
    "CubeData",
    "FilterData",
    "SpectrumData"
]