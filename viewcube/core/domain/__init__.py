"""
Dominio astronómico de ViewCube.

Este módulo principal del dominio astronómico exporta todas las entidades,
modelos y utilidades necesarias para el procesamiento de datos espectrales
astronómicos. Proporciona una interfaz unificada para acceder a todos los
componentes del dominio.

El dominio está organizado en:
- entities: Entidades del negocio astronómico (objetos, instrumentos, coordenadas)
- models: Modelos de datos (espectros, cubos, filtros)

Versión: 2.0.0
"""

# Importar todos los modelos de datos
from .models import (
    # Modelos principales
    SpectrumData,
    CubeData,
    FilterData,
    FilterMetadata,

    # Utilidades de modelos
    DataValidator as ModelsDataValidator,
    CoordinateManager,
    SpectrumExtractor,
    StatisticsCalculator,
    FilterValidator,
    FilterInterpolator,
    FilterCalculator,

    # Excepciones
    FilterError,
    FilterValidationError,
    SpectrumCompatibilityError,

    # Funciones de conveniencia
    create_spectrum,
    create_cube,
    create_filter
)

# Importar todas las entidades del dominio
from .entities import (
    # Enumeraciones
    InstrumentType,
    FiberType,
    SurveyType,

    # Interfaces
    SpatialEntity,
    SpectralEntity,

    # Entidades principales
    SpatialCoordinate,
    WavelengthRange,
    Spaxel,
    AstronomicalObject,
    Instrument,
    Observatory,
    Survey,

    # Calculadoras y utilidades
    SpectralCalculator,
    RedshiftCalculator,
    AstronomicalEntityFactory,
    DataValidator as EntitiesDataValidator
)

# Versión del dominio completo
__version__ = "2.0.0"

# Exportaciones públicas del dominio
__all__ = [
    # === MODELOS DE DATOS ===
    "SpectrumData",
    "CubeData",
    "FilterData",
    "FilterMetadata",

    # === ENTIDADES DEL DOMINIO ===
    "SpatialCoordinate",
    "WavelengthRange",
    "Spaxel",
    "AstronomicalObject",
    "Instrument",
    "Observatory",
    "Survey",

    # === ENUMERACIONES ===
    "InstrumentType",
    "FiberType",
    "SurveyType",

    # === INTERFACES ===
    "SpatialEntity",
    "SpectralEntity",

    # === UTILIDADES Y CALCULADORAS ===
    "SpectralCalculator",
    "RedshiftCalculator",
    "CoordinateManager",
    "SpectrumExtractor",
    "StatisticsCalculator",
    "FilterValidator",
    "FilterInterpolator",
    "FilterCalculator",

    # === FACTORIES ===
    "AstronomicalEntityFactory",

    # === VALIDADORES ===
    "ModelsDataValidator",
    "EntitiesDataValidator",

    # === EXCEPCIONES ===
    "FilterError",
    "FilterValidationError",
    "SpectrumCompatibilityError",

    # === FUNCIONES DE CONVENIENCIA ===
    "create_spectrum",
    "create_cube",
    "create_filter"
]

# Metadatos del dominio
__email__ = "viewcube@astronomy.org"
__status__ = "Production"
__description__ = "Dominio astronómico completo para ViewCube"

# Información del dominio
DOMAIN_INFO = {
    "name": "ViewCube Astronomical Domain",
    "version": __version__,
    "description": "Dominio para procesamiento de datos espectrales astronómicos",
    "modules": {
        "models": "Modelos de datos espectrales (SpectrumData, CubeData, FilterData)",
        "entities": "Entidades astronómicas (objetos, instrumentos, coordenadas)"
    },
    "capabilities": [
        "Procesamiento de espectros 1D",
        "Manejo de cubos espectrales 3D",
        "Aplicación de filtros fotométricos",
        "Gestión de entidades astronómicas",
        "Cálculos astrofísicos y fotométricos"
    ]
}

# Configuración de logging para el dominio
import logging

logger = logging.getLogger(__name__)
logger.info(f"Dominio astronómico ViewCube v{__version__} cargado correctamente")


# Función de utilidad para obtener información del dominio
def get_domain_info():
    """
    Obtiene información completa sobre el dominio astronómico.

    Returns:
        dict: Información detallada del dominio
    """
    return DOMAIN_INFO.copy()


# Función de validación del dominio
def validate_domain_integrity():
    """
    Valida que todos los componentes del dominio estén correctamente cargados.

    Returns:
        bool: True si el dominio está íntegro

    Raises:
        ImportError: Si faltan componentes críticos
    """
    try:
        # Verificar modelos principales
        assert SpectrumData is not None
        assert CubeData is not None
        assert FilterData is not None

        # Verificar entidades principales
        assert AstronomicalObject is not None
        assert Instrument is not None
        assert Survey is not None

        # Verificar utilidades
        assert SpectralCalculator is not None
        assert FilterValidator is not None

        logger.info("Validación de integridad del dominio completada exitosamente")
        return True

    except AssertionError as e:
        logger.error(f"Error en validación de integridad del dominio: {e}")
        raise ImportError("Componentes críticos del dominio no están disponibles")


# Agregar funciones de utilidad a las exportaciones
__all__.extend(["get_domain_info", "validate_domain_integrity"])