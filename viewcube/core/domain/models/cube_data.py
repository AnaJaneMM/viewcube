import numpy as np

class CubeData:
    """
    Modelo de datos para un cubo espectral 3D (lambda, y, x).
    """
    def __init__(self, data, wavelength=None, error=None, flag=None, meta=None):
        self.data = np.asarray(data)
        self.wavelength = np.asarray(wavelength) if wavelength is not None else np.arange(self.data.shape[0])
        self.error = np.asarray(error) if error is not None else None
        self.flag = np.asarray(flag) if flag is not None else None
        self.meta = meta if meta is not None else {}

        # Dimensiones
        self.n_lambda = self.data.shape[0]
        self.n_y = self.data.shape[1]
        self.n_x = self.data.shape[2]

    def get_spaxel_spectrum(self, x, y):
        """
        Devuelve el espectro (lambda) para un spaxel (x, y).
        """
        if 0 <= x < self.n_x and 0 <= y < self.n_y:
            flux = self.data[:, y, x]
            error = self.error[:, y, x] if self.error is not None else None
            flag = self.flag[:, y, x] if self.flag is not None else None
            return SpectrumData(self.wavelength, flux, error, flag)
        return None

    def get_mean_spectrum(self, mask=None):
        """
        Devuelve el espectro promedio sobre todos los spaxels (opcionalmente usando una máscara booleana 2D).
        """
        if mask is not None:
            mask = np.asarray(mask)
            masked_data = np.where(mask, self.data, np.nan)
            mean_flux = np.nanmean(masked_data, axis=(1, 2))
        else:
            mean_flux = np.nanmean(self.data, axis=(1, 2))
        error = np.nanmean(self.error, axis=(1, 2)) if self.error is not None else None
        flag = np.nanmean(self.flag, axis=(1, 2)) if self.flag is not None else None
        return SpectrumData(self.wavelength, mean_flux, error, flag)

    def get_flux_limits(self):
        """
        Devuelve los límites globales de flux en el cubo.
        """
        valid = self.data[np.isfinite(self.data)]
        if valid.size == 0:
            return (np.nan, np.nan)
        return (np.min(valid), np.max(valid))

    def as_dict(self):
        """
        Devuelve todos los datos como diccionario.
        """
        return {
            "data": self.data,
            "wavelength": self.wavelength,
            "error": self.error,
            "flag": self.flag,
            "meta": self.meta
        }

# Importar aquí para evitar dependencia circular
from .spectrum_data import SpectrumData