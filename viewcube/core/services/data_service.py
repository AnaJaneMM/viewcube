"""
Servicio de datos para ViewCube.

Este módulo centraliza todas las operaciones relacionadas con la carga,
procesamiento y manipulación de datos astronómicos.
"""

import os
import sys
import numpy as np
import astropy.io.fits as pyfits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from typing import Optional, Dict, Any, Tuple, Union, List
from collections import OrderedDict

from ..domain.models.spectrum_data import SpectrumData
from ..domain.models.cube_data import CubeData
from ..domain.models.filter_data import FilterData
from ..domain.entities.astronomical_entities import (
    AstronomicalObject,
    SpatialCoordinate,
    WavelengthRange,
    Spaxel,
    Survey,
    InstrumentType,
    FiberType
)


class DataService:
    """
    Servicio para el manejo de datos astronómicos.

    Centraliza todas las operaciones de carga, procesamiento y manipulación
    de datos de cubos espectrales, espectros y tablas de posiciones.
    """

    def __init__(self):
        """Inicializa el servicio de datos."""
        self.speed_of_light = 299792.458  # km/s
        self._loaded_files = {}
        self._cache = {}

    def load_fits_file(self,
                       filename: str,
                       exdata: Optional[int] = None,
                       exhdr: int = 0,
                       exwave: Optional[Union[int, str]] = None,
                       exflag: Optional[int] = None,
                       exerror: Optional[int] = None,
                       specaxis: Optional[int] = None,
                       ivar: bool = False,
                       guess: bool = True,
                       **kwargs) -> Dict[str, Any]:
        """
        Carga un archivo FITS y extrae toda la información relevante.

        Args:
            filename: Ruta al archivo FITS
            exdata: Extensión de datos
            exhdr: Extensión de header
            exwave: Extensión de wavelength
            exflag: Extensión de flags
            exerror: Extensión de errores
            specaxis: Eje espectral
            ivar: Si los errores están en formato IVAR
            guess: Detectar automáticamente extensiones
            **kwargs: Parámetros adicionales

        Returns:
            Diccionario con todos los datos cargados
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Archivo "{filename}" no existe')

        try:
            hdul = pyfits.open(filename)
        except Exception as e:
            raise IOError(f'Error abriendo archivo FITS: {e}')

        try:
            # Información básica
            result = {
                'filename': filename,
                'data': None,
                'header': hdul[exhdr].header,
                'primary_header': hdul[0].header,
                'wavelength': None,
                'error': None,
                'flag': None,
                'wave_rest': None,
                'velocity': None,
                'redshift': None,
                'instrument': None,
                'survey': None
            }

            # Extraer metadatos básicos
            self._extract_metadata(result, hdul)

            # Cargar datos principales
            result['data'] = hdul[exdata if exdata is not None else 0].data

            # Auto-detección de extensiones si está habilitada
            if guess:
                self._auto_detect_extensions(result, hdul)

            # Cargar extensiones específicas
            if exerror is not None:
                result['error'] = hdul[exerror].data
            if exflag is not None:
                result['flag'] = hdul[exflag].data
            if exwave is not None:
                result['wavelength'] = hdul[exwave].data

            # Procesar errores IVAR
            if ivar and result['error'] is not None:
                result['error'] = 1.0 / np.sqrt(result['error'])

            # Extraer información de wavelength desde header
            if result['wavelength'] is None:
                self._extract_wavelength_from_header(result, specaxis)

            # Procesar redshift y velocidad
            self._process_redshift_velocity(result)

            # Crear wavelength en reposo si hay redshift
            if result['redshift'] is not None and result['wavelength'] is not None:
                result['wave_rest'] = result['wavelength'] / (1.0 + result['redshift'])

            return result

        finally:
            hdul.close()

    def _extract_metadata(self, result: Dict[str, Any], hdul: pyfits.HDUList) -> None:
        """Extrae metadatos del archivo FITS."""
        header = result['header']
        primary_header = result['primary_header']

        # Información del survey e instrumento
        result['survey'] = primary_header.get('SURVEY', '').strip() if primary_header.get('SURVEY') else None
        result['instrument'] = primary_header.get('INSTRUME', '').strip() if primary_header.get('INSTRUME') else None
        result['califaid'] = header.get('CALIFAID')

        # Información de versión
        result['pycasso_version'] = None
        if 'QVERSION' in primary_header:
            result['pycasso_version'] = 1
        elif 'PYCASSO VERSION' in primary_header:
            result['pycasso_version'] = 2

    def _auto_detect_extensions(self, result: Dict[str, Any], hdul: pyfits.HDUList) -> None:
        """Detecta automáticamente extensiones comunes en archivos FITS."""
        for i in range(1, len(hdul)):
            if 'EXTNAME' not in hdul[i].header:
                continue

            extname = hdul[i].header['EXTNAME'].split()[0].upper()

            # Extensiones estándar
            if extname == 'ERROR' and result['error'] is None:
                result['error'] = hdul[i].data
            elif extname == 'BADPIX' and result['flag'] is None:
                result['flag'] = hdul[i].data.astype(bool)
            elif extname == 'WAVE' and result['wavelength'] is None:
                result['wavelength'] = hdul[i].data
            elif extname == 'ERRWEIGHT':
                result['errorw'] = hdul[i].data
            elif extname == 'FIBCOVER':
                result['fibcover'] = hdul[i].data
            elif extname == 'FLAT':
                result['flat'] = hdul[i].data

            # Extensiones específicas de instrumentos
            self._detect_instrument_extensions(result, hdul[i], extname)

    def _detect_instrument_extensions(self, result: Dict[str, Any], hdu: pyfits.ImageHDU, extname: str) -> None:
        """Detecta extensiones específicas de diferentes instrumentos."""
        # MaNGA
        if result['instrument'] == 'MaNGA':
            if extname == 'FLUX':
                result['data'] = hdu.data
            elif extname == 'IVAR':
                result['error'] = 1.0 / np.sqrt(hdu.data)
            elif extname == 'MASK':
                result['flag'] = hdu.data

        # MUSE
        elif result['instrument'] == 'MUSE':
            if extname == 'DATA':
                result['data'] = hdu.data
                result['header'] = hdu.header
            elif extname == 'STAT':
                result['error'] = hdu.data

        # WEAVE
        elif result['instrument'] and 'WEAVE' in result['instrument']:
            if extname.endswith('_DATA'):
                result['data'] = hdu.data
                result['header'] = hdu.header
            elif extname.endswith('_IVAR'):
                result['error'] = 1.0 / np.sqrt(hdu.data)
            elif extname.endswith('_SENSFUNC'):
                result['sensfunc'] = hdu.data

    def _extract_wavelength_from_header(self, result: Dict[str, Any], specaxis: Optional[int]) -> None:
        """Extrae información de wavelength desde el header FITS."""
        header = result['header']

        # Determinar eje espectral
        if specaxis is None:
            dispaxis = header.get('DISPAXIS')
            if dispaxis is not None:
                specaxis = dispaxis
            else:
                naxis = header.get('NAXIS', 0)
                specaxis = 3 if naxis == 3 else 1

        # Extraer parámetros de wavelength
        nwave = header.get(f'NAXIS{specaxis}')
        crval = header.get(f'CRVAL{specaxis}')
        cdelt = header.get(f'CDELT{specaxis}')

        if crval is not None and cdelt is not None and nwave is not None:
            result['wavelength'] = crval + cdelt * np.arange(nwave)
            result['crval'] = crval
            result['cdelt'] = cdelt

        # Casos especiales para diferentes instrumentos
        self._handle_special_wavelength_cases(result, header, nwave)

    def _handle_special_wavelength_cases(self, result: Dict[str, Any], header: pyfits.Header,
                                         nwave: Optional[int]) -> None:
        """Maneja casos especiales de wavelength para diferentes instrumentos."""
        # MUSE
        if ('CTYPE3' in header and 'AWAV' in header['CTYPE3'] and
                'CD3_3' in header and result['wavelength'] is None):
            crval = header.get('CRVAL3')
            cdelt = header.get('CD3_3')
            if crval and cdelt and nwave:
                result['wavelength'] = crval + cdelt * np.arange(nwave)

        # WEAVE RSS
        if (result['instrument'] and 'WEAVE' in result['instrument'] and
                'CRVAL1' in header and 'CD1_1' in header and
                result['data'] is not None and result['data'].ndim == 2):
            crval = header.get('CRVAL1')
            cdelt = header.get('CD1_1')
            if crval and cdelt and nwave:
                result['wavelength'] = crval + cdelt * np.arange(nwave)

        # Pycasso v2
        if result['pycasso_version'] == 2 and nwave is not None:
            try:
                wcs = WCS(header)
                result['wavelength'] = self._get_wavelength_coordinates(wcs, nwave)
            except:
                pass

    def _get_wavelength_coordinates(self, wcs: WCS, nwave: int) -> np.ndarray:
        """Obtiene coordenadas de wavelength desde WCS."""
        w = wcs.sub([3])
        pix_coords = np.arange(nwave)
        wave_coords = w.wcs_pix2world(pix_coords[:, np.newaxis], 0)
        if w.wcs.cunit[0] == "m":
            wave_coords *= 1e10
        return np.squeeze(wave_coords)

    def _process_redshift_velocity(self, result: Dict[str, Any]) -> None:
        """Procesa información de redshift y velocidad desde el header."""
        header = result['header']

        # Buscar velocidad en diferentes campos
        velocity_keys = ['CZ', 'MED_VEL', 'V500 MED_VEL']
        for key in velocity_keys:
            if key in header:
                result['velocity'] = float(header[key])
                break

        if result['velocity'] is not None:
            result['redshift'] = result['velocity'] / self.speed_of_light

    def create_spectrum_data(self, wavelength: np.ndarray, flux: np.ndarray,
                             error: Optional[np.ndarray] = None,
                             flag: Optional[np.ndarray] = None,
                             metadata: Optional[Dict] = None) -> SpectrumData:
        """
        Crea un objeto SpectrumData.

        Args:
            wavelength: Array de longitudes de onda
            flux: Array de flujo
            error: Array de errores (opcional)
            flag: Array de flags (opcional)
            metadata: Metadatos adicionales

        Returns:
            Objeto SpectrumData
        """
        return SpectrumData(
            wavelength=wavelength,
            flux=flux,
            error=error,
            flag=flag,
            meta=metadata or {}
        )

    def create_cube_data(self, data: np.ndarray,
                         wavelength: Optional[np.ndarray] = None,
                         error: Optional[np.ndarray] = None,
                         flag: Optional[np.ndarray] = None,
                         metadata: Optional[Dict] = None) -> CubeData:
        """
        Crea un objeto CubeData.

        Args:
            data: Array 3D de datos (lambda, y, x)
            wavelength: Array de longitudes de onda
            error: Array de errores
            flag: Array de flags
            metadata: Metadatos adicionales

        Returns:
            Objeto CubeData
        """
        return CubeData(
            data=data,
            wavelength=wavelength,
            error=error,
            flag=flag,
            meta=metadata or {}
        )

    def extract_spectrum_from_cube(self, cube_data: CubeData, x: int, y: int) -> Optional[SpectrumData]:
        """
        Extrae un espectro de un spaxel específico del cubo.

        Args:
            cube_data: Objeto CubeData
            x: Coordenada X del spaxel
            y: Coordenada Y del spaxel

        Returns:
            Objeto SpectrumData del spaxel o None si las coordenadas son inválidas
        """
        return cube_data.get_spaxel_spectrum(x, y)

    def calculate_integrated_spectrum(self, cube_data: CubeData,
                                      mask: Optional[np.ndarray] = None) -> SpectrumData:
        """
        Calcula el espectro integrado de un cubo.

        Args:
            cube_data: Objeto CubeData
            mask: Máscara opcional para seleccionar spaxels

        Returns:
            Espectro integrado como SpectrumData
        """
        return cube_data.get_mean_spectrum(mask)

    def load_position_table(self, filename: str,
                            filetype: str = "C",
                            angle: Optional[float] = None,
                            skycoord: bool = True) -> Dict[str, Any]:
        """
        Carga una tabla de posiciones.

        Args:
            filename: Ruta del archivo
            filetype: Tipo de fibra ('C' circular, 'H' hexagonal)
            angle: Ángulo de rotación en grados
            skycoord: Usar coordenadas de cielo

        Returns:
            Diccionario con información de posiciones
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Archivo de posiciones "{filename}" no existe')

        try:
            # Detectar formato del archivo
            if filename.endswith('.fits'):
                return self._load_fits_position_table(filename, angle, skycoord)
            else:
                return self._load_text_position_table(filename, angle)
        except Exception as e:
            raise IOError(f'Error cargando tabla de posiciones: {e}')

    def _load_text_position_table(self, filename: str, angle: Optional[float]) -> Dict[str, Any]:
        """Carga tabla de posiciones desde archivo de texto."""
        with open(filename, 'r') as f:
            header_line = f.readline().split()
            filetype, xs, ys = header_line[0], float(header_line[1]), float(header_line[2])

            data = np.loadtxt(filename, skiprows=1, unpack=True)
            fiber_id, x, y = data[0], data[1], data[2]

        # Aplicar rotación si se especifica
        if angle is not None:
            x, y = self._rotate_coordinates(x, y, angle)

        return {
            'fiber_type': filetype,
            'fiber_size_x': xs,
            'fiber_size_y': ys,
            'id': fiber_id,
            'x': x,
            'y': y,
            'radius': xs if filetype == 'C' else None
        }

    def _load_fits_position_table(self, filename: str, angle: Optional[float], skycoord: bool) -> Dict[str, Any]:
        """Carga tabla de posiciones desde archivo FITS."""
        try:
            # Intentar cargar como tabla CALIFA
            return self._load_califa_position_table(filename, angle)
        except:
            try:
                # Intentar cargar como tabla MEGARA
                return self._load_megara_position_table(filename, angle)
            except:
                # Intentar cargar como tabla LIFU/WEAVE
                return self._load_lifu_position_table(filename, angle, skycoord)

    def _load_califa_position_table(self, filename: str, angle: Optional[float]) -> Dict[str, Any]:
        """Carga tabla de posiciones CALIFA."""
        try:
            data, header = pyfits.getdata(filename, extname='FIBERS', header=True)
        except:
            data, header = pyfits.getdata(filename, ext=1, header=True)

        filetype = header['FIBSHAPE'].strip()
        xs = header['FIBSIZEX']
        ys = header['FIBSIZEY']

        x = data[header['TTYPE1']]
        y = data[header['TTYPE2']]

        if angle is not None:
            x, y = self._rotate_coordinates(x, y, angle)

        return {
            'fiber_type': filetype,
            'fiber_size_x': xs,
            'fiber_size_y': ys,
            'x': x,
            'y': y,
            'radius': xs if filetype == 'C' else None
        }

    def _load_megara_position_table(self, filename: str, angle: Optional[float]) -> Dict[str, Any]:
        """Carga tabla de posiciones MEGARA."""
        header = pyfits.getheader(filename, extname='FIBERS')

        fibers = {}
        for key in header:
            if key.startswith('FIB') and '_' in key:
                fid = int(key.split('_')[0].replace('FIB', ''))
                if fid not in fibers:
                    fibers[fid] = {}
                param = key.split('_')[-1]
                fibers[fid][param] = header[key]

        # Extraer coordenadas
        ids = list(fibers.keys())
        x = np.array([fibers[fid]['X'] for fid in ids])
        y = np.array([fibers[fid]['Y'] for fid in ids])

        if angle is not None:
            x, y = self._rotate_coordinates(x, y, angle)

        return {
            'fiber_type': 'H',  # MEGARA usa fibras hexagonales
            'id': np.array(ids),
            'x': x,
            'y': y,
            'radius': self._calculate_fiber_radius(x, y)
        }

    def _load_lifu_position_table(self, filename: str, angle: Optional[float], skycoord: bool) -> Dict[str, Any]:
        """Carga tabla de posiciones LIFU/WEAVE."""
        table = Table.read(filename, hdu='FIBTABLE')

        if skycoord:
            try:
                coords = SkyCoord(table['FIBRERA'] * u.deg, table['FIBREDEC'] * u.deg)
                ref_idx = np.argmin(table['XPOSITION'] ** 2 + table['YPOSITION'] ** 2)
                ra, dec = coords.spherical_offsets_to(coords[ref_idx])
                x = ra.to(u.arcsec).value
                y = dec.to(u.arcsec).value
            except:
                x = table['XPOSITION']
                y = table['YPOSITION']
        else:
            x = table['XPOSITION']
            y = table['YPOSITION']

        if angle is not None:
            x, y = self._rotate_coordinates(x, y, angle)

        # Determinar tamaño de fibra basado en el modo
        header = pyfits.getheader(filename, ext=0)
        mode = header.get('OBSMODE', '').strip().upper()
        fiber_size = 2.6 / 2.0 if 'LIFU' in mode else 1.3 / 2.0

        return {
            'fiber_type': 'C',
            'fiber_size_x': fiber_size,
            'x': x,
            'y': y,
            'radius': fiber_size
        }

    def _rotate_coordinates(self, x: np.ndarray, y: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Rota coordenadas por un ángulo dado."""
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        return x_rot, y_rot

    def _calculate_fiber_radius(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calcula el radio de fibra basado en las distancias entre fibras."""
        from scipy.spatial.distance import cdist

        points = np.column_stack([x, y])
        distances = cdist(points, points, 'euclidean')

        # Encontrar la distancia mínima no cero
        distances[distances == 0] = np.inf
        min_distance = np.min(distances)

        return min_distance / 2.0

    def create_astronomical_object(self, name: str,
                                   object_type: str = "GALAXY",
                                   redshift: Optional[float] = None,
                                   velocity: Optional[float] = None,
                                   coordinates: Optional[SpatialCoordinate] = None) -> AstronomicalObject:
        """
        Crea un objeto astronómico.

        Args:
            name: Nombre del objeto
            object_type: Tipo de objeto
            redshift: Corrimiento al rojo
            velocity: Velocidad radial
            coordinates: Coordenadas del objeto

        Returns:
            Objeto AstronomicalObject
        """
        return AstronomicalObject(
            name=name,
            object_type=object_type,
            redshift=redshift,
            velocity=velocity,
            coordinates=coordinates
        )

    def apply_velocity_correction(self, wavelength: np.ndarray, velocity: float) -> np.ndarray:
        """
        Aplica corrección de velocidad a longitudes de onda.

        Args:
            wavelength: Array de longitudes de onda
            velocity: Velocidad en km/s

        Returns:
            Array de longitudes de onda corregidas
        """
        z = velocity / self.speed_of_light
        return wavelength / (1.0 + z)

    def calculate_redshift_from_velocity(self, velocity: float) -> float:
        """
        Calcula redshift desde velocidad.

        Args:
            velocity: Velocidad en km/s

        Returns:
            Redshift
        """
        return velocity / self.speed_of_light

    def calculate_velocity_from_redshift(self, redshift: float) -> float:
        """
        Calcula velocidad desde redshift.

        Args:
            redshift: Corrimiento al rojo

        Returns:
            Velocidad en km/s
        """
        return redshift * self.speed_of_light

    def get_flux_limits(self, data: np.ndarray, percentile: float = 0.1) -> Tuple[float, float]:
        """
        Calcula límites de flujo para visualización.

        Args:
            data: Array de datos
            percentile: Percentil para calcular límites

        Returns:
            Tupla (min, max) de límites de flujo
        """
        valid_data = data[np.isfinite(data)]
        if valid_data.size == 0:
            return (np.nan, np.nan)

        p_low = np.percentile(valid_data, percentile)
        p_high = np.percentile(valid_data, 100 - percentile)

        return (p_low, p_high)

    def mask_flagged_data(self, data: np.ndarray, flags: np.ndarray, flag_value: int = 0) -> np.ndarray:
        """
        Aplica máscara de flags a los datos.

        Args:
            data: Array de datos
            flags: Array de flags
            flag_value: Valor de flag que indica datos válidos

        Returns:
            Array enmascarado
        """
        mask = flags > flag_value
        return np.ma.array(data, mask=mask)

    def save_spectrum(self, spectrum: SpectrumData, filename: str,
                      format: str = 'txt', header: Optional[pyfits.Header] = None) -> None:
        """
        Guarda un espectro a archivo.

        Args:
            spectrum: Objeto SpectrumData
            filename: Nombre del archivo
            format: Formato ('txt' o 'fits')
            header: Header FITS opcional
        """
        if format.lower() == 'txt':
            self._save_spectrum_txt(spectrum, filename)
        elif format.lower() == 'fits':
            self._save_spectrum_fits(spectrum, filename, header)
        else:
            raise ValueError(f"Formato no soportado: {format}")

    def _save_spectrum_txt(self, spectrum: SpectrumData, filename: str) -> None:
        """Guarda espectro en formato texto."""
        data = np.column_stack([spectrum.wavelength, spectrum.flux])
        if spectrum.error is not None:
            data = np.column_stack([data, spectrum.error])

        header_lines = [
            f"# Spectrum data",
            f"# Wavelength range: {spectrum.get_wavelength_range()}",
            f"# Flux range: {spectrum.get_flux_range()}"
        ]

        np.savetxt(filename, data, header='\n'.join(header_lines), fmt='%.6e')

    def _save_spectrum_fits(self, spectrum: SpectrumData, filename: str,
                            header: Optional[pyfits.Header]) -> None:
        """Guarda espectro en formato FITS."""
        # Crear estructura de datos para FITS
        data = spectrum.flux
        if spectrum.error is not None:
            data = np.column_stack([spectrum.flux, spectrum.error])

        hdu = pyfits.PrimaryHDU(data)

        if header is not None:
            hdu.header.update(header)

        # Añadir información de wavelength
        if len(spectrum.wavelength) > 1:
            crval = spectrum.wavelength[0]
            cdelt = spectrum.wavelength[1] - spectrum.wavelength[0]
            hdu.header['CRVAL1'] = crval
            hdu.header['CDELT1'] = cdelt
            hdu.header['CRPIX1'] = 1

        hdu.writeto(filename, overwrite=True)