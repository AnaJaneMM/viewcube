"""Interfaces del núcleo de la aplicación."""

from .service_interfaces import (
    ServiceInterface,
    DataServiceInterface,
    FilterServiceInterface,
    SonificationServiceInterface,
    EventServiceInterface
)

from .presenter_interfaces import (
    PresenterInterface,
    SpectrumPresenterInterface,
    CubePresenterInterface
)

from .repository_interfaces import (
    RepositoryInterface,
    FitsRepositoryInterface,
    ConfigRepositoryInterface
)

__all__ = [
    # Service Interfaces
    "ServiceInterface",
    "DataServiceInterface",
    "FilterServiceInterface",
    "SonificationServiceInterface",
    "EventServiceInterface",
    # Presenter Interfaces
    "PresenterInterface",
    "SpectrumPresenterInterface",
    "CubePresenterInterface",
    # Repository Interfaces
    "RepositoryInterface",
    "FitsRepositoryInterface",
    "ConfigRepositoryInterface"
]
