"""
Paquete de servicios ViewCube: inicialización modular.

Este __init__.py expone las interfaces principales de los servicios,
garantizando una arquitectura modular, baja complejidad y bajo acoplamiento.
"""

from .data_service import (
    DataService,
    FileValidator,
    FitsFileLoader,
    PositionTableLoader,
    DataFactory,
    AstronomicalCalculator,
    DataProcessor,
    FileIOService,
    LoadResult,
)

from .filter_service import (
    FilterService,
    FilterValidator,
    FilterFileLoader,
    FilterDiscovery,
    FilterCalculator,
    FilterProcessor,
    SyntheticFilterFactory,
    FilterType,
    FilterSearchResult,
)

from .event_service import (
    EventService,
    EventType,
    EventData,
    EventHandlerRegistry,
    SystemState,
    PyQtGraphConnectionManager,
    KeyEventProcessor,
    MouseEventProcessor,
    create_event_service,
    EventServiceTester,
    test_event_service,
)

__all__ = [
    # Data services
    "DataService",
    "FileValidator",
    "FitsFileLoader",
    "PositionTableLoader",
    "DataFactory",
    "AstronomicalCalculator",
    "DataProcessor",
    "FileIOService",
    "LoadResult",
    # Filter services
    "FilterService",
    "FilterValidator",
    "FilterFileLoader",
    "FilterDiscovery",
    "FilterCalculator",
    "FilterProcessor",
    "SyntheticFilterFactory",
    "FilterType",
    "FilterSearchResult",
    # Event services
    "EventService",
    "EventType",
    "EventData",
    "EventHandlerRegistry",
    "SystemState",
    "PyQtGraphConnectionManager",
    "KeyEventProcessor",
    "MouseEventProcessor",
    "create_event_service",
    "EventServiceTester",
    "test_event_service",
]