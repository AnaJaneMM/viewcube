"""Gestión de filtros astronómicos."""

import numpy as np
import os
from typing import List, Optional, Tuple
from .utils import lsfiles


class FilterManager:
    """Gestiona filtros astronómicos y operaciones relacionadas."""

    def __init__(self, filter_directory: str = "filters/"):
        self.filter_directory = filter_directory
        self.available_filters: List[str] = []
        self.current_filter_index: int = 0
        self.current_filter_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._load_available_filters()

    def _load_available_filters(self) -> None:
        """Carga lista de filtros disponibles."""
        if self.filter_directory and os.path.exists(self.filter_directory):
            self.available_filters = lsfiles("*txt", self.filter_directory, path=False) or []

    def set_default_filter(self, filter_name: str) -> bool:
        """Establece filtro por defecto."""
        filter_files = lsfiles("*" + filter_name + "*", self.filter_directory)

        if not filter_files:
            if self.available_filters:
                print(f'"{filter_name}" NOT found. Set to "{self.available_filters[0]}"')
                self.current_filter_index = 0
                return False
            return False

        try:
            self.current_filter_index = self.available_filters.index(filter_files[0])
            print(f"Filter: {'.'.join(filter_files[0].split('.')[:-1])}")
            return True
        except ValueError:
            return False

    def get_current_filter(self) -> Optional[str]:
        """Obtiene el filtro actual."""
        if self.available_filters and 0 <= self.current_filter_index < len(self.available_filters):
            return self.available_filters[self.current_filter_index]
        return None

    def next_filter(self) -> str:
        """Cambia al siguiente filtro."""
        if self.available_filters:
            self.current_filter_index = (self.current_filter_index + 1) % len(self.available_filters)
            return self.get_current_filter()
        return None

    def previous_filter(self) -> str:
        """Cambia al filtro anterior."""
        if self.available_filters:
            self.current_filter_index = (self.current_filter_index - 1) % len(self.available_filters)
            return self.get_current_filter()
        return None

    def load_filter_data(self, wavelength_delta: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Carga datos del filtro actual."""
        current_filter = self.get_current_filter()
        if not current_filter:
            return None

        try:
            filter_path = os.path.join(self.filter_directory, current_filter)
            wavelength, transmission = np.loadtxt(filter_path, unpack=True)

            if wavelength_delta != 0.0:
                wavelength += wavelength_delta

            self.current_filter_data = (wavelength, transmission)
            return self.current_filter_data

        except Exception as e:
            print(f"Error loading filter {current_filter}: {e}")
            return None

    def interpolate_filter(self, target_wavelength: np.ndarray) -> np.ndarray:
        """Interpola filtro a longitudes de onda objetivo."""
        if self.current_filter_data is None:
            return np.zeros_like(target_wavelength)

        filter_wl, filter_trans = self.current_filter_data
        return np.interp(target_wavelength, filter_wl, filter_trans)

    def apply_filter_to_spectrum(self, wavelength: np.ndarray,
                                 spectrum: np.ndarray) -> float:
        """Aplica filtro a espectro y calcula flujo integrado."""
        interpolated_filter = self.interpolate_filter(wavelength)

        if spectrum.ndim == 1:
            numerator = np.trapz(wavelength * interpolated_filter * spectrum, wavelength)
            denominator = np.trapz(interpolated_filter * wavelength, wavelength)
        else:
            # Para múltiples espectros
            numerator = np.trapz(wavelength[:, np.newaxis] * interpolated_filter[:, np.newaxis] * spectrum,
                                 wavelength, axis=0)
            denominator = np.trapz(interpolated_filter * wavelength, wavelength)

        return numerator / denominator if denominator != 0 else 0.0
