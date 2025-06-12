"""
Configuration Manager para ViewCube
Gestiona toda la configuración del sistema de forma centralizada
"""
import os
import configparser
import ast
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ViewCubeConfig:
    """Configuración principal de ViewCube"""
    # Extensiones de datos
    exdata: Optional[int] = None
    exhdr: int = 0
    exwave: Optional[int] = None
    exflag: Optional[int] = None
    exerror: Optional[int] = None
    specaxis: Optional[int] = None

    # Directorios
    dfilter: Optional[str] = "filters/"
    dsoni: Optional[str] = None

    # Configuración de visualización
    norm: str = "sqrt"
    default_filter: Optional[str] = None
    mval: float = 0.0
    palpha: float = 0.95
    plw: float = 0.1
    plc: str = "k"

    # Límites y factores
    wlim: Optional[tuple] = None
    flim: Optional[tuple] = None
    fp: float = 1.2

    # Tamaños de ventana
    fig_spaxel_size: tuple = (7.1, 6)
    fig_spectra_size: tuple = (8, 5)
    fig_window_manager: tuple = (5, 5)

    # Constantes físicas
    c: float = 299792.458

    # Configuración de modo
    ref_mode: str = "crpix"
    soni_start: bool = False
    colorbar: bool = True
    masked: bool = True
    vflag: int = 0


class ConfigurationManager:
    """Gestor centralizado de configuración"""

    def __init__(self, config_file: str = "viewcuberc"):
        self.config_file = config_file
        self.config_path = self._get_config_path()
        self._config = ViewCubeConfig()
        self._load_config()

    def _get_config_path(self) -> str:
        """Obtiene la ruta del archivo de configuración"""
        local_config = os.path.join(os.curdir, self.config_file)
        if os.path.exists(local_config):
            return local_config

        home_config = os.path.join(os.environ.get('HOME', ''), '.' + self.config_file)
        return home_config

    def _load_config(self) -> None:
        """Carga la configuración desde archivo"""
        if not os.path.exists(self.config_path):
            return

        try:
            parser = configparser.ConfigParser()
            parser.read(self.config_path)

            if 'VIEWCUBE' in parser:
                for key, value in parser['VIEWCUBE'].items():
                    if hasattr(self._config, key):
                        try:
                            parsed_value = ast.literal_eval(value)
                            setattr(self._config, key, parsed_value)
                        except (ValueError, SyntaxError):
                            setattr(self._config, key, value)
        except Exception as e:
            print(f"Warning: Error loading config: {e}")

    def save_config(self) -> None:
        """Guarda la configuración actual"""
        parser = configparser.ConfigParser()
        parser['VIEWCUBE'] = {}

        for field_name, field_value in self._config.__dict__.items():
            parser['VIEWCUBE'][field_name] = str(field_value)

        with open(self.config_path, 'w') as f:
            parser.write(f)

    def get_config(self) -> ViewCubeConfig:
        """Obtiene la configuración actual"""
        return self._config

    def update_config(self, **kwargs) -> None:
        """Actualiza la configuración con nuevos valores"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def create_default_config(self) -> None:
        """Crea un archivo de configuración por defecto"""
        self.save_config()
        print(f'*** ViewCube Config File "{self.config_path}" created! ***')
