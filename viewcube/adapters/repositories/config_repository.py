import os
import yaml
from typing import Dict, Any
from pathlib import Path
from ...core.interfaces.repository_interfaces import ConfigRepositoryInterface


class ConfigRepository(ConfigRepositoryInterface):
    """Implementación concreta del repositorio de configuración"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(Path.home() / '.viewcube' / 'config.yml')
        self.default_config = {
            'display': {
                'colormap': 'viridis',
                'norm': 'sqrt',
                'default_filter': 'Halpha_KPNO-NOAO'
            },
            'data': {
                'fits': {
                    'auto_detect': True,
                    'guess_extensions': True
                }
            }
        }

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        path = config_path or self.config_path
        if not os.path.exists(path):
            return self.default_config

        with open(path, 'r') as f:
            user_config = yaml.safe_load(f) or {}

        return self.merge_config(self.default_config, user_config)

    def save_config(self, config: Dict[str, Any], config_path: Optional[str] = None) -> None:
        path = config_path or self.config_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            yaml.safe_dump(config, f)

    def get_default_config(self) -> Dict[str, Any]:
        return self.default_config

    def merge_config(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Combina configuraciones usando deep merge"""
        merged = default_config.copy()

        for key, value in user_config.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self.merge_config(merged[key], value)
            else:
                merged[key] = value

        return merged

    def update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """Actualización recursiva de diccionarios anidados"""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self.update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d