"""
Core del sistema ViewCube.

Este módulo centraliza los componentes principales del sistema, incluyendo
modelos de dominio, interfaces y servicios, siguiendo los principios SOLID
y arquitectura limpia para garantizar bajo acoplamiento y alta cohesión.

Versión: 2.0.0
"""

import logging
import sys
from typing import Dict, List, Any, Optional, Type, TypeVar, Set

# Configuración de logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Metadatos del módulo
__version__ = '2.0.0'
__license__ = 'MIT'

# Variables de control
_initialized = False
_registered_components = set()
_component_registry = {}

# Tipo genérico para componentes
T = TypeVar('T')


def initialize() -> bool:
    """
    Inicializa el core del sistema, cargando todos los componentes necesarios.

    Returns:
        bool: True si la inicialización fue exitosa, False en caso contrario
    """
    global _initialized

    if _initialized:
        logger.warning("El core ya ha sido inicializado")
        return True

    try:
        # Inicializar modelos de dominio
        _initialize_domain_models()

        # Inicializar interfaces
        _initialize_interfaces()

        # Inicializar servicios
        _initialize_services()

        _initialized = True
        logger.info(f"Core inicializado correctamente (v{__version__})")
        return True
    except Exception as e:
        logger.error(f"Error inicializando el core: {e}")
        return False


def _initialize_domain_models() -> None:
    """
    Inicializa los modelos de dominio del sistema.
    """
    try:
        # Importar modelos
        from .domain.models.spectrum_data import SpectrumData
        from .domain.models.cube_data import CubeData
        from .domain.models.filter_data import FilterData

        # Importar entidades
        from .domain.entities.astronomical_entities import (
            AstronomicalObject, Spaxel, SpatialCoordinate, WavelengthRange,
            InstrumentType, FiberType, SurveyType
        )

        # Registrar modelos
        _register_component(SpectrumData, 'models')
        _register_component(CubeData, 'models')
        _register_component(FilterData, 'models')

        # Registrar entidades
        _register_component(AstronomicalObject, 'entities')
        _register_component(Spaxel, 'entities')
        _register_component(SpatialCoordinate, 'entities')
        _register_component(WavelengthRange, 'entities')
        _register_component(InstrumentType, 'entities')
        _register_component(FiberType, 'entities')
        _register_component(SurveyType, 'entities')

        logger.debug("Modelos de dominio inicializados correctamente")
    except ImportError as e:
        logger.error(f"Error importando modelos de dominio: {e}")
        raise


def _initialize_interfaces() -> None:
    """
    Inicializa las interfaces del sistema.
    """
    try:
        # Importar interfaces de presentadores
        from .interfaces.presenter_interfaces import (
            PresenterInterface, SpectrumPresenterInterface, CubePresenterInterface,
            PyQtGraphPresenterMixin, InteractivePresenterInterface
        )

        # Importar interfaces de repositorios
        from .interfaces.repository_interfaces import (
            RepositoryInterface, FitsRepositoryInterface, ConfigRepositoryInterface,
            FilterRepositoryInterface, CacheRepositoryInterface
        )

        # Importar interfaces de servicios
        from .interfaces.service_interfaces import (
            ServiceInterface, DataServiceInterface, FilterServiceInterface,
            SonificationServiceInterface, EventServiceInterface, ValidationServiceInterface,
            CacheServiceInterface, CalculationServiceInterface
        )

        # Registrar interfaces de presentadores
        _register_component(PresenterInterface, 'interfaces')
        _register_component(SpectrumPresenterInterface, 'interfaces')
        _register_component(CubePresenterInterface, 'interfaces')
        _register_component(PyQtGraphPresenterMixin, 'interfaces')
        _register_component(InteractivePresenterInterface, 'interfaces')

        # Registrar interfaces de repositorios
        _register_component(RepositoryInterface, 'interfaces')
        _register_component(FitsRepositoryInterface, 'interfaces')
        _register_component(ConfigRepositoryInterface, 'interfaces')
        _register_component(FilterRepositoryInterface, 'interfaces')
        _register_component(CacheRepositoryInterface, 'interfaces')

        # Registrar interfaces de servicios
        _register_component(ServiceInterface, 'interfaces')
        _register_component(DataServiceInterface, 'interfaces')
        _register_component(FilterServiceInterface, 'interfaces')
        _register_component(SonificationServiceInterface, 'interfaces')
        _register_component(EventServiceInterface, 'interfaces')
        _register_component(ValidationServiceInterface, 'interfaces')
        _register_component(CacheServiceInterface, 'interfaces')
        _register_component(CalculationServiceInterface, 'interfaces')

        logger.debug("Interfaces inicializadas correctamente")
    except ImportError as e:
        logger.error(f"Error importando interfaces: {e}")
        raise


def _initialize_services() -> None:
    """
    Inicializa los servicios del sistema.
    """
    try:
        # Importar servicios
        from .services.data_service import DataService
        from .services.event_service import EventService
        from .services.filter_service import FilterService
        from .services.sonification_service import SonificationService

        # Registrar servicios
        _register_component(DataService, 'services')
        _register_component(EventService, 'services')
        _register_component(FilterService, 'services')
        _register_component(SonificationService, 'services')

        logger.debug("Servicios inicializados correctamente")
    except ImportError as e:
        logger.error(f"Error importando servicios: {e}")
        raise


def _register_component(component: Type[T], category: str) -> None:
    """
    Registra un componente en el registro del sistema.

    Args:
        component: Clase del componente a registrar
        category: Categoría del componente (models, entities, interfaces, services)
    """
    component_name = component.__name__
    if component_name in _registered_components:
        logger.warning(f"Componente {component_name} ya registrado")
        return

    if category not in _component_registry:
        _component_registry[category] = {}

    _component_registry[category][component_name] = component
    _registered_components.add(component_name)
    logger.debug(f"Componente {component_name} registrado en categoría {category}")


def get_component(name: str) -> Optional[Type]:
    """
    Obtiene un componente por su nombre.

    Args:
        name: Nombre del componente

    Returns:
        Clase del componente o None si no existe
    """
    for category in _component_registry.values():
        if name in category:
            return category[name]
    return None


def get_components_by_category(category: str) -> Dict[str, Type]:
    """
    Obtiene todos los componentes de una categoría.

    Args:
        category: Categoría de componentes (models, entities, interfaces, services)

    Returns:
        Diccionario con los componentes de la categoría
    """
    return _component_registry.get(category, {}).copy()


def get_all_components() -> Dict[str, Dict[str, Type]]:
    """
    Obtiene todos los componentes registrados.

    Returns:
        Diccionario con todos los componentes organizados por categoría
    """
    return _component_registry.copy()


def get_version() -> str:
    """
    Obtiene la versión actual del core.

    Returns:
        Versión del core
    """
    return __version__


def is_initialized() -> bool:
    """
    Verifica si el core ha sido inicializado.

    Returns:
        True si el core está inicializado, False en caso contrario
    """
    return _initialized


# Exportar componentes principales para facilitar su importación
from .domain.models.spectrum_data import SpectrumData
from .domain.models.cube_data import CubeData
from .domain.models.filter_data import FilterData
from .domain.entities.astronomical_entities import (
    AstronomicalObject, Spaxel, SpatialCoordinate, WavelengthRange
)
from .interfaces.presenter_interfaces import SpectrumPresenterInterface, CubePresenterInterface
from .interfaces.repository_interfaces import FitsRepositoryInterface, ConfigRepositoryInterface
from .interfaces.service_interfaces import DataServiceInterface, FilterServiceInterface
from .services.data_service import DataService
from .services.event_service import EventService
from .services.filter_service import FilterService
from .services.sonification_service import SonificationService

# Inicializar automáticamente si se importa el módulo
if not _initialized:
    initialize()