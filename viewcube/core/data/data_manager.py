"""Gestión centralizada de datos astronómicos."""

import numpy as np
from typing import Optional, Tuple, Any
from .utils import LoadFits
from abc import ABC, abstractmethod


class DataManagerInterface(ABC):
    """Interfaz para gestores de datos."""

    @abstractmethod
    def load_primary_data(self, filename: str, **kwargs) -> 'SpectrumData':
        pass

    @abstractmethod
    def load_comparison_data(self, filename: str, **kwargs) -> Optional['SpectrumData']:
        pass


class SpectrumData:
    """Encapsula datos espectrales con validación."""

    def __init__(self, data: np.ndarray, wavelength: np.ndarray,
                 header: dict, error: Optional[np.ndarray] = None,
                 flag: Optional[np.ndarray] = None):
        self._validate_data(data, wavelength)
        self.data = data
        self.wavelength = wavelength
        self.header = header
        self.error = error
        self.flag = flag
        self._setup_metadata()

    def _validate_data(self, data: np.ndarray, wavelength: np.ndarray) -> None:
        """Valida consistencia de datos."""
        if data is None or wavelength is None:
            raise ValueError("Data and wavelength cannot be None")

        if data.ndim not in [2, 3]:
            raise ValueError("Data must be 2D or 3D array")

        expected_wave_len = data.shape[0] if data.ndim == 2 else data.shape[0]
        if len(wavelength) != expected_wave_len:
            raise ValueError("Wavelength array length mismatch")

    def _setup_metadata(self) -> None:
        """Configura metadatos derivados."""
        self.shape = self.data.shape
        self.ndim = self.data.ndim

        if self.ndim == 2:
            self.n_spectra, self.n_wavelength = self.shape
        else:
            self.n_wavelength, self.y_size, self.x_size = self.shape

    def get_spectrum(self, index: int) -> np.ndarray:
        """Obtiene un espectro específico."""
        if self.ndim == 2:
            return self.data[index, :]
        else:
            raise NotImplementedError("3D spectrum extraction needs coordinates")

    def get_spectrum_3d(self, x: int, y: int) -> np.ndarray:
        """Obtiene espectro de posición específica en cubo 3D."""
        if self.ndim != 3:
            raise ValueError("Only valid for 3D data")
        return self.data[:, y, x]


class DataManager(DataManagerInterface):
    """Gestor principal de datos astronómicos."""

    def __init__(self):
        self.primary_data: Optional[SpectrumData] = None
        self.comparison_data: Optional[SpectrumData] = None
        self._cached_filters = {}

    def load_primary_data(self, filename: str, **kwargs) -> SpectrumData:
        """Carga datos principales desde archivo FITS."""
        try:
            fobj = LoadFits(filename, **kwargs)

            data = fobj.data
            wavelength = fobj.wave
            header = fobj.hdr
            error = fobj.error
            flag = fobj.flag

            self.primary_data = SpectrumData(data, wavelength, header, error, flag)
            self.primary_data.fobj = fobj  # Mantener referencia para compatibilidad

            return self.primary_data

        except Exception as e:
            raise RuntimeError(f"Error loading primary data from {filename}: {e}")

    def load_comparison_data(self, filename: str, **kwargs) -> Optional[SpectrumData]:
        """Carga datos de comparación desde archivo FITS."""
        if not filename:
            return None

        try:
            fobj = LoadFits(filename, **kwargs)

            data = fobj.data
            wavelength = fobj.wave
            header = fobj.hdr
            error = fobj.error
            flag = fobj.flag

            self.comparison_data = SpectrumData(data, wavelength, header, error, flag)
            self.comparison_data.fobj = fobj

            return self.comparison_data

        except Exception as e:
            print(f"Warning: Could not load comparison data from {filename}: {e}")
            return None

    def apply_multiplicative_factors(self, primary_factor: float = 1.0,
                                     comparison_factor: float = 1.0) -> None:
        """Aplica factores multiplicativos a los datos."""
        if self.primary_data and primary_factor != 1.0:
            self.primary_data.data *= primary_factor
            if self.primary_data.error is not None:
                self.primary_data.error *= primary_factor

        if self.comparison_data and comparison_factor != 1.0:
            self.comparison_data.data *= comparison_factor
            if self.comparison_data.error is not None:
                self.comparison_data.error *= comparison_factor

    def get_wavelength_limits(self) -> Tuple[float, float]:
        """Obtiene límites de longitud de onda combinados."""
        wl_min = self.primary_data.wavelength.min()
        wl_max = self.primary_data.wavelength.max()

        if self.comparison_data:
            wl_min = min(wl_min, self.comparison_data.wavelength.min())
            wl_max = max(wl_max, self.comparison_data.wavelength.max())

        return wl_min, wl_max