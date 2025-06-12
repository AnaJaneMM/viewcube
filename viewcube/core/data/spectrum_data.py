"""Clase encapsuladora de datos espectrales con validación."""

import numpy as np
from typing import Optional, Tuple, Any


class SpectrumData:
    """Encapsula datos espectrales con validación y metadatos."""

    def __init__(self, data: np.ndarray, wavelength: np.ndarray,
                 header: dict, error: Optional[np.ndarray] = None,
                 flag: Optional[np.ndarray] = None, syn: Optional[np.ndarray] = None,
                 fobj: Optional[Any] = None):
        """
        Inicializa el contenedor de datos espectrales.

        Parameters:
        -----------
        data : np.ndarray
            Array de datos espectrales (2D o 3D)
        wavelength : np.ndarray
            Array de longitudes de onda
        header : dict
            Encabezado FITS
        error : Optional[np.ndarray]
            Array de errores
        flag : Optional[np.ndarray]
            Array de flags/máscaras
        syn : Optional[np.ndarray]
            Datos sintéticos (para PyCasso)
        fobj : Optional[Any]
            Objeto LoadFits original para compatibilidad
        """
        self._validate_data(data, wavelength)

        self.data = data
        self.wavelength = wavelength
        self.header = header
        self.error = error
        self.flag = flag
        self.syn = syn
        self.fobj = fobj

        self._setup_metadata()

    def _validate_data(self, data: np.ndarray, wavelength: np.ndarray) -> None:
        """Valida consistencia de datos."""
        if data is None or wavelength is None:
            raise ValueError("Data and wavelength cannot be None")

        if data.ndim not in [2, 3]:
            raise ValueError("Data must be 2D or 3D array")

        # Para datos 2D: (n_spectra, n_wavelength)
        # Para datos 3D: (n_wavelength, y_size, x_size)
        expected_wave_len = data.shape[1] if data.ndim == 2 else data.shape[0]
        if len(wavelength) != expected_wave_len:
            raise ValueError(f"Wavelength array length ({len(wavelength)}) "
                             f"doesn't match data shape ({expected_wave_len})")

    def _setup_metadata(self) -> None:
        """Configura metadatos derivados."""
        self.shape = self.data.shape
        self.ndim = self.data.ndim

        if self.ndim == 2:
            self.n_spectra, self.n_wavelength = self.shape
            self.x_size = None
            self.y_size = None
        else:  # 3D
            self.n_wavelength, self.y_size, self.x_size = self.shape
            self.n_spectra = self.y_size * self.x_size

    def get_spectrum(self, index: int) -> np.ndarray:
        """Obtiene un espectro específico por índice (para datos 2D)."""
        if self.ndim != 2:
            raise ValueError("get_spectrum() only valid for 2D data. Use get_spectrum_3d() for 3D data")

        if index < 0 or index >= self.n_spectra:
            raise IndexError(f"Spectrum index {index} out of range [0, {self.n_spectra - 1}]")

        return self.data[index, :]

    def get_spectrum_3d(self, x: int, y: int) -> np.ndarray:
        """Obtiene espectro de posición específica en cubo 3D."""
        if self.ndim != 3:
            raise ValueError("get_spectrum_3d() only valid for 3D data")

        if x < 0 or x >= self.x_size or y < 0 or y >= self.y_size:
            raise IndexError(f"Coordinates ({x}, {y}) out of range")

        return self.data[:, y, x]

    def get_error_spectrum(self, x: int = None, y: int = None, index: int = None) -> Optional[np.ndarray]:
        """Obtiene espectro de errores."""
        if self.error is None:
            return None

        if self.ndim == 2:
            if index is None:
                raise ValueError("Index required for 2D data")
            return self.error[index, :]
        else:
            if x is None or y is None:
                raise ValueError("x and y coordinates required for 3D data")
            return self.error[:, y, x]

    def get_flag_spectrum(self, x: int = None, y: int = None, index: int = None) -> Optional[np.ndarray]:
        """Obtiene espectro de flags."""
        if self.flag is None:
            return None

        if self.ndim == 2:
            if index is None:
                raise ValueError("Index required for 2D data")
            return self.flag[index, :]
        else:
            if x is None or y is None:
                raise ValueError("x and y coordinates required for 3D data")
            return self.flag[:, y, x]

    def get_synthetic_spectrum(self, x: int = None, y: int = None, index: int = None) -> Optional[np.ndarray]:
        """Obtiene espectro sintético (PyCasso)."""
        if self.syn is None:
            return None

        if self.ndim == 2:
            if index is None:
                raise ValueError("Index required for 2D data")
            return self.syn[index, :]
        else:
            if x is None or y is None:
                raise ValueError("x and y coordinates required for 3D data")
            return self.syn[:, y, x]

    def integrate_spectra(self, indices: list = None, coordinates: list = None) -> np.ndarray:
        """Integra múltiples espectros."""
        if self.ndim == 2:
            if indices is None:
                raise ValueError("Indices required for 2D data")
            return np.sum(self.data[indices, :], axis=0)
        else:
            if coordinates is None:
                raise ValueError("Coordinates required for 3D data")
            integrated = np.zeros(self.n_wavelength)
            for x, y in coordinates:
                integrated += self.data[:, y, x]
            return integrated

    def __repr__(self) -> str:
        return (f"SpectrumData(shape={self.shape}, ndim={self.ndim}, "
                f"has_error={self.error is not None}, "
                f"has_flag={self.flag is not None}, "
                f"has_synthetic={self.syn is not None})")
