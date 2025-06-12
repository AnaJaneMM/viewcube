"""Módulo de gestión de datos."""

from .data_manager import DataManager, DataManagerInterface
from .spectrum_data import SpectrumData
from .filter_manager import FilterManager

__all__ = ["DataManager", "DataManagerInterface", "SpectrumData", "FilterManager"]
