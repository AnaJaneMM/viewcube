"""Interfaz base abstracta para visualizadores."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class ViewerInterface(ABC):
    """Interfaz abstracta base para todos los visualizadores."""

    @abstractmethod
    def setup_figure(self) -> None:
        """Configura la figura y ejes principales."""
        pass

    @abstractmethod
    def update_display(self) -> None:
        """Actualiza la visualización."""
        pass

    @abstractmethod
    def clear_display(self) -> None:
        """Limpia la visualización."""
        pass

    @abstractmethod
    def set_title(self, title: str) -> None:
        """Establece el título del visualizador."""
        pass
