"""
Interfaces principales del sistema ViewCube refactorizado.

Este módulo centraliza todas las interfaces del core siguiendo principios de
arquitectura limpia, responsabilidad única y bajo acoplamiento.

Versión: 2.0.0
Arquitectura: Clean Architecture con PyQt5/PyQtGraph
Compatibilidad: Python 3.8+
"""

import logging
from typing import TYPE_CHECKING

# Configuración de logging para el módulo de interfaces
logger = logging.getLogger(__name__)

# Version information
__version__ = "2.0.0"
__email__ = "viewcube@astronomy.org"
__status__ = "Production"

# Importaciones principales de interfaces de repositorio
try:
    from .repository_interfaces import (
        RepositoryInterface,
        FitsRepositoryInterface,
        ConfigRepositoryInterface,
        FilterRepositoryInterface,
        CacheRepositoryInterface
    )

    _repository_imports_success = True
    logger.debug("Repository interfaces imported successfully")
except ImportError as e:
    logger.error(f"Failed to import repository interfaces: {e}")
    _repository_imports_success = False
    raise

# Importaciones principales de interfaces de servicios
try:
    from .service_interfaces import (
        ServiceInterface,
        DataServiceInterface,
        FilterServiceInterface,
        SonificationServiceInterface,
        EventServiceInterface,
        ValidationServiceInterface,
        CacheServiceInterface,
        CalculationServiceInterface
    )

    _service_imports_success = True
    logger.debug("Service interfaces imported successfully")
except ImportError as e:
    logger.error(f"Failed to import service interfaces: {e}")
    _service_imports_success = False
    raise

# Importaciones principales de interfaces de presentadores
try:
    from .presenter_interfaces import (
        PresenterInterface,
        SpectrumPresenterInterface,
        CubePresenterInterface,
        PyQtGraphPresenterMixin,
        InteractivePresenterInterface
    )

    _presenter_imports_success = True
    logger.debug("Presenter interfaces imported successfully")
except ImportError as e:
    logger.error(f"Failed to import presenter interfaces: {e}")
    _presenter_imports_success = False
    raise


# Verificación de importaciones exitosas
def _verify_imports():
    """
    Verifica que todas las importaciones críticas se hayan realizado correctamente.

    Returns:
        bool: True si todas las importaciones fueron exitosas

    Raises:
        ImportError: Si alguna importación crítica falló
    """
    if not all([_repository_imports_success, _service_imports_success, _presenter_imports_success]):
        missing_modules = []
        if not _repository_imports_success:
            missing_modules.append("repository_interfaces")
        if not _service_imports_success:
            missing_modules.append("service_interfaces")
        if not _presenter_imports_success:
            missing_modules.append("presenter_interfaces")

        error_msg = f"Critical interface modules failed to import: {missing_modules}"
        logger.critical(error_msg)
        raise ImportError(error_msg)

    logger.info("All interface modules imported successfully")
    return True


# Ejecutar verificación al importar el módulo
_verify_imports()

# Interfaces principales exportadas (compatibilidad con versión anterior)
__all__ = [
    # Interfaces de repositorio principales
    "FitsRepositoryInterface",
    "ConfigRepositoryInterface",

    # Interfaces de servicios principales
    "DataServiceInterface",
    "FilterServiceInterface",

    # Interfaces de presentadores principales
    "SpectrumPresenterInterface",
    "CubePresenterInterface",

    # Interfaces base para extensibilidad
    "RepositoryInterface",
    "ServiceInterface",
    "PresenterInterface",

    # Interfaces especializadas para funcionalidades avanzadas
    "SonificationServiceInterface",
    "EventServiceInterface",
    "ValidationServiceInterface",
    "CacheServiceInterface",
    "CalculationServiceInterface",
    "FilterRepositoryInterface",
    "CacheRepositoryInterface",
    "PyQtGraphPresenterMixin",
    "InteractivePresenterInterface",

    # Metadatos del módulo
    "__version__",
    "__author__",
    "__email__",
    "__status__"
]

# Diccionario de categorización de interfaces para facilitar su uso
INTERFACE_CATEGORIES = {
    "repository": [
        "RepositoryInterface",
        "FitsRepositoryInterface",
        "ConfigRepositoryInterface",
        "FilterRepositoryInterface",
        "CacheRepositoryInterface"
    ],
    "service": [
        "ServiceInterface",
        "DataServiceInterface",
        "FilterServiceInterface",
        "SonificationServiceInterface",
        "EventServiceInterface",
        "ValidationServiceInterface",
        "CacheServiceInterface",
        "CalculationServiceInterface"
    ],
    "presenter": [
        "PresenterInterface",
        "SpectrumPresenterInterface",
        "CubePresenterInterface",
        "PyQtGraphPresenterMixin",
        "InteractivePresenterInterface"
    ]
}


# Funciones de utilidad para facilitar el uso del módulo
def get_interfaces_by_category(category: str) -> list:
    """
    Obtiene todas las interfaces de una categoría específica.

    Args:
        category: Categoría de interfaces ('repository', 'service', 'presenter')

    Returns:
        Lista de nombres de interfaces en la categoría

    Raises:
        ValueError: Si la categoría no existe
    """
    if category not in INTERFACE_CATEGORIES:
        available_categories = list(INTERFACE_CATEGORIES.keys())
        raise ValueError(f"Category '{category}' not found. Available: {available_categories}")

    return INTERFACE_CATEGORIES[category].copy()


def list_all_interfaces() -> dict:
    """
    Lista todas las interfaces disponibles organizadas por categoría.

    Returns:
        Diccionario con todas las interfaces organizadas por categoría
    """
    return INTERFACE_CATEGORIES.copy()


def get_interface_info() -> dict:
    """
    Obtiene información completa del módulo de interfaces.

    Returns:
        Diccionario con metadatos e información del módulo
    """
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "status": __status__,
        "total_interfaces": len(__all__) - 4,  # Excluyendo metadatos
        "categories": list(INTERFACE_CATEGORIES.keys()),
        "interfaces_by_category": INTERFACE_CATEGORIES
    }


# Validación de integridad del módulo
def _validate_module_integrity():
    """
    Valida la integridad del módulo verificando que todas las interfaces
    declaradas estén realmente disponibles.

    Returns:
        bool: True si la validación es exitosa

    Raises:
        RuntimeError: Si hay inconsistencias en el módulo
    """
    # Verificar que todas las interfaces en __all__ existan en el namespace
    current_module = globals()
    missing_interfaces = []

    for interface_name in __all__:
        if interface_name.startswith("__"):  # Skip metadata
            continue
        if interface_name not in current_module:
            missing_interfaces.append(interface_name)

    if missing_interfaces:
        error_msg = f"Interfaces declared in __all__ but not available: {missing_interfaces}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Verificar que todas las interfaces categorizadas estén en __all__
    categorized_interfaces = set()
    for interfaces in INTERFACE_CATEGORIES.values():
        categorized_interfaces.update(interfaces)

    declared_interfaces = set(i for i in __all__ if not i.startswith("__"))

    if categorized_interfaces != declared_interfaces:
        missing_in_all = categorized_interfaces - declared_interfaces
        extra_in_all = declared_interfaces - categorized_interfaces

        if missing_in_all:
            logger.warning(f"Interfaces in categories but not in __all__: {missing_in_all}")
        if extra_in_all:
            logger.warning(f"Interfaces in __all__ but not categorized: {extra_in_all}")

    logger.info("Module integrity validation passed")
    return True


# Ejecutar validación de integridad
_validate_module_integrity()


# Inicialización del módulo
def _initialize_module():
    """
    Inicializa el módulo de interfaces con configuración básica.
    """
    logger.info(f"ViewCube Core Interfaces v{__version__} initialized successfully")
    logger.info(f"Available interface categories: {list(INTERFACE_CATEGORIES.keys())}")
    logger.info(f"Total interfaces available: {len(__all__) - 4}")


# Ejecutar inicialización
_initialize_module()

# Compatibilidad con type checking
if TYPE_CHECKING:
    # Importaciones adicionales solo para type checking
    from typing import Protocol, runtime_checkable

# Cleanup de variables internas
del _repository_imports_success, _service_imports_success, _presenter_imports_success
del logger  # El logger se mantiene solo durante la inicialización