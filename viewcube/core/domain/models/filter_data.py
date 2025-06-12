import numpy as np

class FilterData:
    """
    Modelo de datos para un filtro espectral.
    """
    def __init__(self, wavelength, response, name=None, meta=None):
        self.wavelength = np.asarray(wavelength)
        self.response = np.asarray(response)
        self.name = name
        self.meta = meta if meta is not None else {}

    def apply_to_spectrum(self, spectrum: SpectrumData):
        """
        Aplica la respuesta del filtro a un espectro y devuelve el espectro filtrado.
        """
        interp_response = np.interp(spectrum.wavelength, self.wavelength, self.response)
        filtered_flux = spectrum.flux * interp_response
        return SpectrumData(spectrum.wavelength, filtered_flux, spectrum.error, spectrum.flag, spectrum.meta)

    def integrate_flux(self, spectrum: SpectrumData):
        """
        Calcula la integración del espectro bajo el filtro (flujo total transmitido).
        """
        interp_response = np.interp(spectrum.wavelength, self.wavelength, self.response)
        return np.trapz(spectrum.flux * interp_response, spectrum.wavelength)

    def as_dict(self):
        """
        Devuelve todos los datos como diccionario.
        """
        return {
            "wavelength": self.wavelength,
            "response": self.response,
            "name": self.name,
            "meta": self.meta
        }

# Importar aquí para evitar dependencia circular
from .spectrum_data import SpectrumData