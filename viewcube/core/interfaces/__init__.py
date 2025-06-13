"""Interfaces principales del sistema."""
from .repository_interfaces import FitsRepositoryInterface, ConfigRepositoryInterface
from .service_interfaces import DataServiceInterface, FilterServiceInterface
from .presenter_interfaces import SpectrumPresenterInterface, CubePresenterInterface

__all__ = [
    "FitsRepositoryInterface",
    "ConfigRepositoryInterface",
    "DataServiceInterface",
    "FilterServiceInterface",
    "SpectrumPresenterInterface",
    "CubePresenterInterface"
]