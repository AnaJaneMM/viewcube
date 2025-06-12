"""Núcleo de la aplicación ViewCube."""

# Importar solo las interfaces públicas principales
from .domain import (
    AstronomicalObject,
    SpectrumData,
    CubeData,
    FilterData
)

from .services import (
    DataService,
    FilterService,
    SonificationService,
    EventService
)

# Interfaces principales para extensibilidad
from .interfaces import (
    ServiceInterface,
    PresenterInterface,
    RepositoryInterface
)

__all__ = [
    # Domain Objects
    "AstronomicalObject",
    "SpectrumData",
    "CubeData",
    "FilterData",
    # Services
    "DataService",
    "FilterService",
    "SonificationService",
    "EventService",
    # Main Interfaces
    "ServiceInterface",
    "PresenterInterface",
    "RepositoryInterface"
]