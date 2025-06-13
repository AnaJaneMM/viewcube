"""
Modelos de datos del dominio astronómico.

Este módulo exporta los modelos de datos fundamentales para el manejo
de información espectral astronómica, incluyendo espectros individuales,
cubos de datos 3D y filtros espectrales.

Versión: 2.0.0
"""

from .spectrum_data import SpectrumData
from .cube_data import (
    CubeData,
    DataValidator,
    CoordinateManager,
    SpectrumExtractor,
    StatisticsCalculator
)
from .filter_data import (
    FilterData,
    FilterMetadata,
    FilterValidator,
    FilterInterpolator,
    FilterCalculator,
    FilterError,
    FilterValidationError,
    SpectrumCompatibilityError
)

# Versión del módulo de modelos
__version__ = "2.0.0"

# Exportaciones públicas principales
__all__ = [
    # Modelos principales
    "SpectrumData",
    "CubeData",
    "FilterData",

    # Metadatos y configuración
    "FilterMetadata",

    # Validadores y utilidades de CubeData
    "DataValidator",
    "CoordinateManager",
    "SpectrumExtractor",
    "StatisticsCalculator",

    # Validadores y utilidades de FilterData
    "FilterValidator",
    "FilterInterpolator",
    "FilterCalculator",

    # Excepciones específicas
    "FilterError",
    "FilterValidationError",
    "SpectrumCompatibilityError"
]

# Metadatos del módulo
__email__ = "viewcube@astronomy.org"
__status__ = "Production"
__description__ = "Modelos de datos para procesamiento espectroscópico astronómico"

# Configuración de logging para modelos
import logging

logger = logging.getLogger(__name__)
logger.info("Módulo de modelos de datos espectrales cargado correctamente")


# Funciones de conveniencia para crear instancias
def create_spectrum(wavelength, flux, **kwargs):
    """
    Función de conveniencia para crear un objeto SpectrumData.

    Args:
        wavelength: Array de longitudes de onda
        flux: Array de flujo
        **kwargs: Argumentos adicionales para SpectrumData

    Returns:
        SpectrumData: Instancia del espectro
    """
    return SpectrumData(wavelength, flux, **kwargs)


def create_cube(data, **kwargs):
    """
    Función de conveniencia para crear un objeto CubeData.

    Args:
        data: Array 3D de datos espectrales
        **kwargs: Argumentos adicionales para CubeData

    Returns:
        CubeData: Instancia del cubo de datos
    """
    return CubeData(data, **kwargs)


def create_filter(wavelength, response, **kwargs):
    """
    Función de conveniencia para crear un objeto FilterData.

    Args:
        wavelength: Array de longitudes de onda del filtro
        response: Array de respuesta del filtro
        **kwargs: Argumentos adicionales para FilterData

    Returns:
        FilterData: Instancia del filtro
    """
    return FilterData(wavelength, response, **kwargs)


# Agregar funciones de conveniencia a las exportaciones
__all__.extend(["create_spectrum", "create_cube", "create_filter"])