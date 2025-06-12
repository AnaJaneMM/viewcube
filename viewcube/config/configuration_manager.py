"""Gestor de configuración global del sistema ViewCube."""

import os
import json
import configparser
from typing import Any, Dict, Optional, Union
from collections import OrderedDict


class ConfigurationManager:
    """Gestor de configuración global para ViewCube."""

    def __init__(self, config_file: str = None):
        """
        Inicializa el gestor de configuración.

        Parameters:
        -----------
        config_file : str
            Archivo de configuración personalizado
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = OrderedDict()
        self._setup_default_config()
        self.load_config()

    def _get_default_config_path(self) -> str:
        """Obtiene la ruta por defecto del archivo de configuración."""
        home_dir = os.path.expanduser("~")
        return os.path.join(home_dir, ".viewcuberc")

    def _setup_default_config(self) -> None:
        """Configura los parámetros por defecto."""
        self.config = OrderedDict([
            # Extensiones de datos
            ("exdata", None),
            ("exhdr", 0),
            ("exwave", None),
            ("exflag", None),
            ("exerror", None),
            ("specaxis", None),

            # Directorios
            ("dfilter", "filters/"),
            ("dsoni", None),

            # Configuración de visualización
            ("ref_mode", "crpix"),
            ("norm", "sqrt"),
            ("default_filter", "Halpha_KPNO-NOAO"),
            ("mval", 0.0),
            ("palpha", 0.95),
            ("plw", 0.1),
            ("plc", "k"),
            ("clw", 1),
            ("cc", "r"),
            ("cf", False),
            ("ca", 0.8),
            ("slw", 2),
            ("sf", False),
            ("sa", 0.9),

            # Colores de espectros
            ("cspec", "#1f77b4"),
            ("lspec", 1),
            ("ccom", "#ff7f0e"),
            ("lcom", 1),
            ("cflag", "r"),
            ("lflag", 1),

            # Opciones de visualización
            ("colorbar", True),
            ("fits", False),
            ("txt", True),
            ("integrated", True),
            ("individual", False),
            ("wlim", None),
            ("flim", None),
            ("iclm", True),
            ("fp", 1.2),

            # Tamaños de ventanas
            ("fig_spaxel_size", (7.1, 6)),
            ("fig_spectra_size", (8, 5)),
            ("fig_window_manager", (5, 5)),

            # Constantes físicas
            ("c", 299792.458),

            # Opciones avanzadas
            ("cfilter", False),
            ("remove_cont", False),
            ("angle", None),
            ("skycoord", True),
            ("masked", True),
            ("vflag", 0),
            ("soni_start", False),
        ])

    def load_config(self) -> None:
        """Carga configuración desde archivo."""
        if not os.path.exists(self.config_file):
            return

        try:
            if self.config_file.endswith('.json'):
                self._load_json_config()
            else:
                self._load_ini_config()
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")

    def _load_json_config(self) -> None:
        """Carga configuración desde archivo JSON."""
        with open(self.config_file, 'r') as f:
            file_config = json.load(f)

        for key, value in file_config.items():
            if key in self.config:
                self.config[key] = value

    def _load_ini_config(self) -> None:
        """Carga configuración desde archivo INI."""
        parser = configparser.ConfigParser()
        parser.read(self.config_file)

        if 'VIEWCUBE' in parser:
            for key, value in parser['VIEWCUBE'].items():
                if key in self.config:
                    # Intentar evaluar el valor como Python literal
                    try:
                        self.config[key] = eval(value)
                    except:
                        self.config[key] = value

    def save_config(self, format_type: str = 'ini') -> None:
        """
        Guarda configuración a archivo.

        Parameters:
        -----------
        format_type : str
            Formato del archivo: 'ini' o 'json'
        """
        try:
            if format_type == 'json':
                self._save_json_config()
            else:
                self._save_ini_config()
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config file: {e}")

    def _save_json_config(self) -> None:
        """Guarda configuración en formato JSON."""
        # Convertir tuplas a listas para serialización JSON
        config_to_save = {}
        for key, value in self.config.items():
            if isinstance(value, tuple):
                config_to_save[key] = list(value)
            else:
                config_to_save[key] = value

        with open(self.config_file, 'w') as f:
            json.dump(config_to_save, f, indent=2)

    def _save_ini_config(self) -> None:
        """Guarda configuración en formato INI."""
        parser = configparser.ConfigParser()
        parser.add_section('VIEWCUBE')

        for key, value in self.config.items():
            parser.set('VIEWCUBE', key, str(value))

        with open(self.config_file, 'w') as f:
            f.write(f"[VIEWCUBE] # ViewCube Configuration File\n")
            for key, value in self.config.items():
                if value is not None:
                    comment = self._get_parameter_comment(key)
                    f.write(f"# {comment}\n")
                    f.write(f"{key} = {value}\n\n")

    def _get_parameter_comment(self, key: str) -> str:
        """Obtiene comentario descriptivo para un parámetro."""
        comments = {
            "exdata": "Extension Data",
            "exhdr": "Extension Header",
            "exwave": "External Wavelength",
            "exflag": "FLAG/MASK Extension",
            "exerror": "ERROR Extension",
            "specaxis": "Spectral Dimension",
            "dfilter": "Filters directory",
            "dsoni": "Sonification directory",
            "ref_mode": "Reference (central) pixel [crpix | max]",
            "norm": "Normalization for colorbar",
            "default_filter": "Default filter",
            "mval": "Masked values",
            "palpha": "Alpha channel value",
            "plw": "Line width",
            "plc": "Color",
            "clw": "Line width selection",
            "cc": "Color selection",
            "cf": "Fill selection",
            "ca": "Alpha selection",
            "slw": "Selection line width",
            "sf": "Selection fill",
            "sa": "Selection alpha",
            "cspec": "Spectra color",
            "lspec": "Spectra linewidth",
            "ccom": "Comparison spectra color",
            "lcom": "Comparison spectra linewidth",
            "cflag": "Flag color",
            "lflag": "Flag linewidth",
            "colorbar": "Show colorbar",
            "fits": "Save FITS format",
            "txt": "Save ASCII format",
            "integrated": "Save integrated spectrum",
            "individual": "Save individual spectra",
            "wlim": "Wavelength limits",
            "flim": "Flux limits",
            "iclm": "Interactive colormap limits",
            "fp": "Filter scaling factor",
            "fig_spaxel_size": "Spaxel viewer figure size",
            "fig_spectra_size": "Spectra viewer figure size",
            "fig_window_manager": "Window manager figure size",
            "c": "Speed of light (km/s)",
            "cfilter": "Center filter in wavelength range",
            "remove_cont": "Remove continuum",
            "angle": "Rotation angle for position table",
            "skycoord": "Use sky coordinates for distance computation",
            "masked": "Use masked arrays for flagged values",
            "vflag": "Flag threshold value",
            "soni_start": "Start sonification mode"
        }
        return comments.get(key, f"Parameter: {key}")

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Obtiene valor de un parámetro.

        Parameters:
        -----------
        key : str
            Nombre del parámetro
        default : Any
            Valor por defecto si no existe

        Returns:
        --------
        Any
            Valor del parámetro
        """
        return self.config.get(key, default)

    def set_parameter(self, key: str, value: Any) -> None:
        """
        Establece valor de un parámetro.

        Parameters:
        -----------
        key : str
            Nombre del parámetro
        value : Any
            Nuevo valor
        """
        self.config[key] = value

    def get_all_parameters(self) -> Dict[str, Any]:
        """Obtiene todos los parámetros como diccionario."""
        return dict(self.config)

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Actualiza múltiples parámetros.

        Parameters:
        -----------
        parameters : Dict[str, Any]
            Diccionario con parámetros a actualizar
        """
        for key, value in parameters.items():
            if key in self.config:
                self.config[key] = value

    def reset_to_defaults(self) -> None:
        """Resetea configuración a valores por defecto."""
        self._setup_default_config()

    def create_config_file(self, force: bool = False) -> None:
        """
        Crea archivo de configuración con valores por defecto.

        Parameters:
        -----------
        force : bool
            Forzar creación aunque el archivo exista
        """
        if os.path.exists(self.config_file) and not force:
            print(f"Config file {self.config_file} already exists. Use force=True to overwrite.")
            return

        self.save_config()

    def __getitem__(self, key: str) -> Any:
        """Permite acceso como diccionario."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Permite asignación como diccionario."""
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """Verifica si existe un parámetro."""
        return key in self.config

    def keys(self):
        """Obtiene las claves de configuración."""
        return self.config.keys()

    def values(self):
        """Obtiene los valores de configuración."""
        return self.config.values()

    def items(self):
        """Obtiene pares clave-valor de configuración."""
        return self.config.items()
