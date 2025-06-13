import os
import numpy as np
import astropy.io.fits as pyfits
from astropy.wcs import WCS
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from ...core.interfaces.repository_interfaces import FitsRepositoryInterface


class FitsRepository(FitsRepositoryInterface):
    """Implementación concreta del repositorio FITS siguiendo Clean Architecture"""

    def __init__(self):
        self._cache = {}
        self.speed_of_light = 299792.458  # km/s

    def load_fits_file(self, filename: str, **kwargs) -> Dict[str, Any]:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Archivo FITS no encontrado: {filename}")

        try:
            hdul = pyfits.open(filename)
            result = self._process_hdul(hdul, **kwargs)
            return result
        except Exception as e:
            raise IOError(f"Error leyendo archivo FITS: {e}")
        finally:
            hdul.close()

    def _process_hdul(self, hdul, **kwargs):
        exhdr = kwargs.get('exhdr', 0)
        exdata = kwargs.get('exdata')
        exwave = kwargs.get('exwave')
        exflag = kwargs.get('exflag')
        exerror = kwargs.get('exerror')
        specaxis = kwargs.get('specaxis')
        guess = kwargs.get('guess', True)
        ivar = kwargs.get('ivar', False)

        result = {
            'filename': Path(hdul.filename()).name,
            'data': None,
            'header': hdul[exhdr].header,
            'primary_header': hdul[0].header,
            'wavelength': None,
            'error': None,
            'flag': None,
            'velocity': None,
            'redshift': None,
            'instrument': None,
            'survey': None
        }

        self._extract_metadata(result, hdul)
        self._load_main_data(result, hdul, exdata, guess)
        self._process_errors_flags(result, hdul, exerror, exflag, ivar)
        self._process_wavelength(result, hdul, exwave, specaxis)
        self._process_redshift_velocity(result)

        return result

    def _extract_metadata(self, result: Dict, hdul: pyfits.HDUList):
        header = result['header']
        primary_header = result['primary_header']

        result['survey'] = primary_header.get('SURVEY')
        result['instrument'] = primary_header.get('INSTRUME')
        result['califaid'] = header.get('CALIFAID')

        if 'QVERSION' in primary_header:
            result['pycasso_version'] = 1
        elif 'PYCASSO VERSION' in primary_header:
            result['pycasso_version'] = 2

    def _load_main_data(self, result: Dict, hdul, exdata: Optional[int], guess: bool):
        if exdata is not None:
            result['data'] = hdul[exdata].data
        elif guess:
            self._auto_detect_data(result, hdul)

    def _auto_detect_data(self, result: Dict, hdul: pyfits.HDUList):
        for i in range(1, len(hdul)):
            if 'EXTNAME' not in hdul[i].header:
                continue

            extname = hdul[i].header['EXTNAME'].upper()
            if extname in ['FLUX', 'DATA']:
                result['data'] = hdul[i].data
                result['header'] = hdul[i].header
                break

    def _process_errors_flags(self, result: Dict, hdul, exerror: Optional[int], exflag: Optional[int], ivar: bool):
        if exerror is not None:
            result['error'] = hdul[exerror].data
        if exflag is not None:
            result['flag'] = hdul[exflag].data.astype(bool)

        if ivar and result['error'] is not None:
            result['error'] = 1.0 / np.sqrt(result['error'])

    def _process_wavelength(self, result: Dict, hdul, exwave, specaxis: Optional[int]):
        if exwave is not None:
            result['wavelength'] = hdul[exwave].data
        else:
            self._extract_wavelength_from_header(result, specaxis)

        if result['redshift'] and result['wavelength']:
            result['wave_rest'] = result['wavelength'] / (1 + result['redshift'])

    def extract_wavelength(self, header: Dict, specaxis: Optional[int] = None) -> Optional[np.ndarray]:
        specaxis = specaxis or header.get('DISPAXIS') or (3 if header.get('NAXIS') == 3 else 1)

        crval = header.get(f'CRVAL{specaxis}')
        cdelt = header.get(f'CDELT{specaxis}')
        nwave = header.get(f'NAXIS{specaxis}')

        if all([crval, cdelt, nwave]):
            return crval + cdelt * np.arange(nwave)
        return None

    def save_spectrum(self, wavelength: np.ndarray, flux: np.ndarray,
                      filename: str, header: Optional[Dict] = None) -> None:
        primary_hdu = pyfits.PrimaryHDU()
        hdul = pyfits.HDUList([primary_hdu])

        col1 = pyfits.Column(name='WAVE', format='D', array=wavelength)
        col2 = pyfits.Column(name='FLUX', format='D', array=flux)
        table_hdu = pyfits.BinTableHDU.from_columns([col1, col2])

        if header:
            table_hdu.header.update(header)

        hdul.append(table_hdu)
        hdul.writeto(filename, overwrite=True)

    def _process_redshift_velocity(self, result: Dict):
        header = result['header']
        velocity_keys = ['CZ', 'MED_VEL', 'V500 MED_VEL']

        for key in velocity_keys:
            if key in header:
                result['velocity'] = float(header[key])
                break

        if result['velocity']:
            result['redshift'] = result['velocity'] / self.speed_of_light