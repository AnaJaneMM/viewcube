import numpy as np

class SpectrumData:
    """
    Modelo de datos para un espectro individual.
    """
    def __init__(self, wavelength, flux, error=None, flag=None, meta=None):
        self.wavelength = np.asarray(wavelength)
        self.flux = np.asarray(flux)
        self.error = np.asarray(error) if error is not None else None
        self.flag = np.asarray(flag) if flag is not None else None
        self.meta = meta if meta is not None else {}

    def apply_flag_mask(self, mask_value=0):
        """
        Aplica una máscara a los datos de flux utilizando la flag.
        """
        if self.flag is not None:
            mask = self.flag > mask_value
            self.flux = np.ma.array(self.flux, mask=mask)
            if self.error is not None:
                self.error = np.ma.array(self.error, mask=mask)

    def get_flux_range(self):
        """
        Devuelve el rango (min, max) de flux ignorando NaNs y máscaras.
        """
        if np.ma.isMaskedArray(self.flux):
            valid = self.flux.compressed()
        else:
            valid = self.flux[np.isfinite(self.flux)]
        if valid.size == 0:
            return (np.nan, np.nan)
        return (np.min(valid), np.max(valid))

    def get_wavelength_range(self):
        """
        Devuelve el rango (min, max) de longitud de onda.
        """
        valid = self.wavelength[np.isfinite(self.wavelength)]
        if valid.size == 0:
            return (np.nan, np.nan)
        return (np.min(valid), np.max(valid))

    def as_dict(self):
        """
        Devuelve todos los datos como diccionario.
        """
        return {
            "wavelength": self.wavelength,
            "flux": self.flux,
            "error": self.error,
            "flag": self.flag,
            "meta": self.meta
        }