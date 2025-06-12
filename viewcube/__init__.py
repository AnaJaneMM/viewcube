"""ViewCube: Visualizador de datos astronómicos refactorizado."""

__version__ = "1.0.0"
__author__ = "RGB@IAA"

from viewcube.ui.controllers import CubeController
from .config.configuration_manager import ConfigurationManager

__all__ = ["CubeController", "ConfigurationManager"]