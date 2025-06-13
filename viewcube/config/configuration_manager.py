import os
import yaml
from pathlib import Path
from typing import Dict, Any
from ..core.interfaces.repository_interfaces import ConfigRepositoryInterface


class ConfigurationManager:
    """Gestor centralizado de configuración del sistema"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        """Inicialización interna del gestor"""
        self._default_config = {
            'display': {
                'colormap': 'viridis',
                'norm': 'sqrt',
                'default_filter': 'Halpha_KPNO-NOAO',
                'figsize': {'width': 12, 'height': 8}
            },
            'data': {
                'fits': {
                    'auto_detect': True,
                    'guess_extensions': True,
                    'supported_instruments': ['CALIFA', 'MANGA', 'MUSE', 'WEAVE']
                }
            },
            'performance': {
                'cache_size': '1GB',
                'parallel_processing': False
            }
        }
        self._user_config = {}
        self._config_path = Path.home() / '.viewcube' / 'config.yml'
        self._repo = ConfigRepositoryInterface()

    def load_config(self) -> Dict[str, Any]:
        """Carga la configuración combinando valores por defecto y usuario"""
        if not os.path.exists(self._config_path):
            return self._default_config

        user_config = self._repo.load_config(str(self._config_path))
        return self._deep_merge(self._default_config, user_config)

    def save_config(self, config: Dict[str, Any]) -> None:
        """Guarda la configuración del usuario"""
        self._repo.save_config(config, str(self._config_path))

    def get_default_config(self) -> Dict[str, Any]:
        """Devuelve la configuración por defecto"""
        return self._default_config.copy()

    def get_current_config(self) -> Dict[str, Any]:
        """Devuelve la configuración actual en uso"""
        return self.load_config()

    def reset_to_defaults(self) -> None:
        """Restaura la configuración a los valores por defecto"""
        if os.path.exists(self._config_path):
            os.remove(self._config_path)
        self._user_config = {}

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Fusión recursiva de diccionarios"""
        for key, value in update.items():
            if isinstance(value, dict):
                node = base.setdefault(key, {})
                self._deep_merge(node, value)
            else:
                base[key] = value
        return base

    def get(self, key: str, default=None) -> Any:
        """Obtiene un valor de configuración por clave"""
        keys = key.split('.')
        current = self.load_config()

        for k in keys:
            current = current.get(k, {})
            if not isinstance(current, dict):
                break

        return current if current is not {} else default

    def set(self, key: str, value: Any) -> None:
        """Establece un valor de configuración"""
        keys = key.split('.')
        current = self._user_config

        for k in keys[:-1]:
            current = current.setdefault(k, {})

        current[keys[-1]] = value
        self.save_config(self._user_config)