"""
Servicio de datos mejorado para ViewCube.

Esta refactorización mejora la documentación, el manejo de errores
y la consistencia de interfaces, manteniendo toda la funcionalidad original.
"""

import os
import sys
import numpy as np
import astropy.io.fits as pyfits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from typing import Optional, Dict, Any, Tuple, Union, List, Protocol
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from pathlib import Path

# Configurar logging
logger = logging.getLogger(__name__)


class DataLoaderProtocol(Protocol):
    """Protocolo para cargadores de datos."""

    def load(self, source: str, **kwargs) -> Dict[str, Any]:
        """Carga datos desde una fuente."""
        ...


class DataProcessorProtocol(Protocol):
    """Protocolo para procesadores de datos."""

    def process(self, data: Any, **kwargs) -> Any:
        """Procesa datos."""
        ...


class ValidationProtocol(Protocol):
    """Protocolo para validadores de datos."""

    def validate(self, data: Any) -> bool:
        """Valida datos."""
        ...


@dataclass
class LoadResult:
    """
    Resultado de carga de datos FITS con validación mejorada.

    Attributes:
        data: Array de datos principales
        header: Header FITS de datos
        primary_header: Header primario del archivo
        wavelength: Array de longitudes de onda (opcional)
        error: Array de errores (opcional)
        flag: Array de flags (opcional)
        metadata: Metadatos adicionales
    """
    data: np.ndarray
    header: pyfits.Header
    primary_header: pyfits.Header
    wavelength: Optional[np.ndarray] = None
    error: Optional[np.ndarray] = None
    flag: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Inicialización post-creación con validación."""
        if self.metadata is None:
            self.metadata = {}
        self._validate_data()

    def _validate_data(self) -> None:
        """Valida la consistencia de los datos cargados."""
        if self.data is None:
            raise ValueError("Los datos principales no pueden ser None")

        if self.wavelength is not None and len(self.wavelength) != self.data.shape[0]:
            logger.warning("Longitud de wavelength no coincide con datos principales")

        if self.error is not None and self.error.shape != self.data.shape:
            logger.warning("Forma de array de errores no coincide con datos principales")


class FileValidator:
    """Validador especializado para archivos FITS."""

    @staticmethod
    def validate_fits_file(filename: str) -> bool:
        """
        Valida que un archivo FITS sea accesible y tenga estructura válida.

        Args:
            filename: Ruta al archivo FITS

        Returns:
            True si el archivo es válido

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el archivo no es un FITS válido
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Archivo "{filename}" no existe')

        try:
            with pyfits.open(filename) as hdul:
                if len(hdul) == 0:
                    raise ValueError("Archivo FITS vacío")
                return True
        except Exception as e:
            raise ValueError(f"Archivo FITS inválido: {e}")


class FitsFileLoader:
    """
    Servicio especializado para cargar archivos FITS con validación mejorada.

    Aplica el principio de responsabilidad única centralizando
    toda la lógica relacionada con la carga de archivos FITS.
    """

    def __init__(self, validator: Optional[FileValidator] = None):
        """
        Inicializa el cargador de archivos FITS.

        Args:
            validator: Validador de archivos (opcional)
        """
        self.speed_of_light = 299792.458  # km/s
        self._validator = validator or FileValidator()
        self._metadata_extractor = MetadataExtractor()
        self._extension_detector = ExtensionDetector()
        self._wavelength_extractor = WavelengthExtractor()

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
                       **kwargs) -> LoadResult:
        """
        Carga un archivo FITS completo con validación mejorada.

        Args:
            filename: Ruta al archivo FITS
            exdata: Extensión de datos principales
            exhdr: Extensión del header principal
            exwave: Extensión de wavelength
            exflag: Extensión de flags
            exerror: Extensión de errores
            specaxis: Eje espectral
            ivar: Si los errores están en formato IVAR
            guess: Detectar automáticamente extensiones
            **kwargs: Parámetros adicionales

        Returns:
            LoadResult con todos los datos cargados

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el archivo FITS es inválido
            IOError: Si hay problemas al abrir el archivo
        """
        # Validar archivo antes de procesar
        self._validator.validate_fits_file(filename)

        try:
            with pyfits.open(filename) as hdul:
                result = self._create_base_result(hdul, exdata, exhdr, filename)

                if guess:
                    self._extension_detector.auto_detect_extensions(result, hdul)

                self._load_specific_extensions(result, hdul, exerror, exflag, exwave)

                if ivar and result.error is not None:
                    result.error = self._convert_ivar_to_error(result.error)

                if result.wavelength is None:
                    result.wavelength = self._wavelength_extractor.extract_from_header(
                        result.header, specaxis, result.data.shape if result.data is not None else None
                    )

                self._process_redshift_velocity(result)

                if result.metadata.get('redshift') and result.wavelength is not None:
                    result.metadata['wave_rest'] = self._calculate_rest_wavelength(
                        result.wavelength, result.metadata['redshift']
                    )

                logger.info(f"Archivo FITS cargado exitosamente: {filename}")
                return result

        except Exception as e:
            logger.error(f"Error cargando archivo FITS {filename}: {e}")
            raise IOError(f'Error abriendo archivo FITS: {e}')

    def _create_base_result(self, hdul: pyfits.HDUList,
                            exdata: Optional[int], exhdr: int, filename: str) -> LoadResult:
        """
        Crea el resultado base con datos principales y validación.

        Args:
            hdul: Lista de HDUs del archivo FITS
            exdata: Extensión de datos
            exhdr: Extensión de header
            filename: Nombre del archivo para metadatos

        Returns:
            LoadResult con datos base
        """
        try:
            header = hdul[exhdr].header
            primary_header = hdul[0].header
            data = hdul[exdata if exdata is not None else 0].data

            metadata = self._metadata_extractor.extract_metadata(header, primary_header)
            metadata['filename'] = filename
            metadata['load_timestamp'] = time.time()

            return LoadResult(
                data=data,
                header=header,
                primary_header=primary_header,
                metadata=metadata
            )
        except IndexError as e:
            raise ValueError(f"Extensión FITS inválida: {e}")
        except Exception as e:
            raise IOError(f"Error creando resultado base: {e}")

    def _load_specific_extensions(self, result: LoadResult, hdul: pyfits.HDUList,
                                  exerror: Optional[int], exflag: Optional[int],
                                  exwave: Optional[Union[int, str]]) -> None:
        """
        Carga extensiones específicas con manejo de errores mejorado.

        Args:
            result: Resultado a modificar
            hdul: Lista de HDUs
            exerror: Extensión de errores
            exflag: Extensión de flags
            exwave: Extensión de wavelength
        """
        try:
            if exerror is not None and exerror < len(hdul):
                result.error = hdul[exerror].data

            if exflag is not None and exflag < len(hdul):
                result.flag = hdul[exflag].data

            if exwave is not None and exwave < len(hdul):
                result.wavelength = hdul[exwave].data
        except Exception as e:
            logger.warning(f"Error cargando extensiones específicas: {e}")

    def _convert_ivar_to_error(self, ivar_data: np.ndarray) -> np.ndarray:
        """
        Convierte datos IVAR a errores con validación.

        Args:
            ivar_data: Array de inverse variance

        Returns:
            Array de errores convertidos
        """
        if not isinstance(ivar_data, np.ndarray):
            raise TypeError("IVAR data debe ser un numpy array")

        # Evitar división por cero
        safe_ivar = np.maximum(ivar_data, np.finfo(float).eps)
        return 1.0 / np.sqrt(safe_ivar)

    def _process_redshift_velocity(self, result: LoadResult) -> None:
        """
        Procesa información de redshift y velocidad con validación.

        Args:
            result: Resultado a modificar
        """
        velocity_keys = ['CZ', 'MED_VEL', 'V500', 'MED_VEL']

        for key in velocity_keys:
            if key in result.header:
                try:
                    velocity = float(result.header[key])
                    result.metadata['velocity'] = velocity
                    result.metadata['redshift'] = velocity / self.speed_of_light
                    break
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error procesando velocidad de clave {key}: {e}")

    def _calculate_rest_wavelength(self, wavelength: np.ndarray, redshift: float) -> np.ndarray:
        """
        Calcula wavelength en reposo con validación.

        Args:
            wavelength: Array de longitudes de onda observadas
            redshift: Corrimiento al rojo

        Returns:
            Array de longitudes de onda en reposo
        """
        if redshift < -1:
            logger.warning(f"Redshift inválido: {redshift}")
            return wavelength

        return wavelength / (1.0 + redshift)


class MetadataExtractor:
    """Extractor especializado de metadatos FITS con validación mejorada."""

    def extract_metadata(self, header: pyfits.Header,
                         primary_header: pyfits.Header) -> Dict[str, Any]:
        """
        Extrae metadatos básicos de headers FITS con validación.

        Args:
            header: Header principal de datos
            primary_header: Header primario del archivo

        Returns:
            Diccionario con metadatos extraídos y validados
        """
        metadata = {}

        # Información del survey e instrumento
        survey = self._safe_extract_string(primary_header, 'SURVEY')
        if survey:
            metadata['survey'] = survey

        instrument = self._safe_extract_string(primary_header, 'INSTRUME')
        if instrument:
            metadata['instrument'] = instrument

        # ID de CALIFA
        califaid = self._safe_extract_value(header, 'CALIFAID')
        if califaid:
            metadata['califaid'] = califaid

        # Versión de Pycasso
        if 'QVERSION' in primary_header:
            metadata['pycasso_version'] = 1
        elif 'PYCASSO VERSION' in primary_header:
            metadata['pycasso_version'] = 2

        return metadata

    def _safe_extract_string(self, header: pyfits.Header, key: str) -> Optional[str]:
        """
        Extrae de forma segura un valor string del header.

        Args:
            header: Header FITS
            key: Clave a extraer

        Returns:
            Valor extraído o None si no existe/es inválido
        """
        try:
            value = header.get(key)
            return value.strip() if value and isinstance(value, str) else None
        except Exception:
            return None

    def _safe_extract_value(self, header: pyfits.Header, key: str) -> Optional[Any]:
        """
        Extrae de forma segura cualquier valor del header.

        Args:
            header: Header FITS
            key: Clave a extraer

        Returns:
            Valor extraído o None si no existe
        """
        try:
            return header.get(key)
        except Exception:
            return None


class ExtensionDetector:
    """Detector automático de extensiones FITS con mejor manejo de errores."""

    def auto_detect_extensions(self, result: LoadResult, hdul: pyfits.HDUList) -> None:
        """
        Detecta automáticamente extensiones comunes con validación.

        Args:
            result: Resultado de carga a modificar
            hdul: Lista de HDUs del archivo FITS
        """
        for i in range(1, len(hdul)):
            try:
                if 'EXTNAME' not in hdul[i].header:
                    continue

                extname = hdul[i].header['EXTNAME']
                if not isinstance(extname, str):
                    continue

                extname = extname.split()[0].upper()

                self._process_standard_extension(result, hdul[i], extname)
                self._process_instrument_extension(result, hdul[i], extname)

            except Exception as e:
                logger.warning(f"Error procesando extensión {i}: {e}")

    def _process_standard_extension(self, result: LoadResult,
                                    hdu: pyfits.ImageHDU, extname: str) -> None:
        """
        Procesa extensiones estándar con validación.

        Args:
            result: Resultado a modificar
            hdu: HDU a procesar
            extname: Nombre de la extensión
        """
        try:
            if extname == 'ERROR' and result.error is None:
                result.error = hdu.data
            elif extname == 'BADPIX' and result.flag is None:
                result.flag = hdu.data.astype(bool)
            elif extname == 'WAVE' and result.wavelength is None:
                result.wavelength = hdu.data
            elif extname == 'ERRWEIGHT':
                result.metadata['errorw'] = hdu.data
            elif extname == 'FIBCOVER':
                result.metadata['fibcover'] = hdu.data
            elif extname == 'FLAT':
                result.metadata['flat'] = hdu.data
        except Exception as e:
            logger.warning(f"Error procesando extensión estándar {extname}: {e}")

    def _process_instrument_extension(self, result: LoadResult,
                                      hdu: pyfits.ImageHDU, extname: str) -> None:
        """
        Procesa extensiones específicas de instrumentos con validación.

        Args:
            result: Resultado a modificar
            hdu: HDU a procesar
            extname: Nombre de la extensión
        """
        instrument = result.metadata.get('instrument', '')

        try:
            if instrument == 'MaNGA':
                self._process_manga_extension(result, hdu, extname)
            elif instrument == 'MUSE':
                self._process_muse_extension(result, hdu, extname)
            elif 'WEAVE' in instrument:
                self._process_weave_extension(result, hdu, extname)
        except Exception as e:
            logger.warning(f"Error procesando extensión de {instrument}: {e}")

    def _process_manga_extension(self, result: LoadResult,
                                 hdu: pyfits.ImageHDU, extname: str) -> None:
        """Procesa extensiones específicas de MaNGA con validación."""
        if extname == 'FLUX':
            result.data = hdu.data
        elif extname == 'IVAR' and hdu.data is not None:
            result.error = 1.0 / np.sqrt(np.maximum(hdu.data, np.finfo(float).eps))
        elif extname == 'MASK':
            result.flag = hdu.data

    def _process_muse_extension(self, result: LoadResult,
                                hdu: pyfits.ImageHDU, extname: str) -> None:
        """Procesa extensiones específicas de MUSE con validación."""
        if extname == 'DATA':
            result.data = hdu.data
            result.header = hdu.header
        elif extname == 'STAT':
            result.error = hdu.data

    def _process_weave_extension(self, result: LoadResult,
                                 hdu: pyfits.ImageHDU, extname: str) -> None:
        """Procesa extensiones específicas de WEAVE con validación."""
        if extname.endswith('_DATA'):
            result.data = hdu.data
            result.header = hdu.header
        elif extname.endswith('_IVAR') and hdu.data is not None:
            result.error = 1.0 / np.sqrt(np.maximum(hdu.data, np.finfo(float).eps))
        elif extname.endswith('_SENSFUNC'):
            result.metadata['sensfunc'] = hdu.data


class WavelengthExtractor:
    """Extractor especializado de información de wavelength con mejor validación."""

    def extract_from_header(self, header: pyfits.Header,
                            specaxis: Optional[int],
                            data_shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        """
        Extrae wavelength desde header FITS con validación mejorada.

        Args:
            header: Header FITS
            specaxis: Eje espectral
            data_shape: Forma de los datos para determinar dimensiones

        Returns:
            Array de wavelengths o None si no se puede extraer
        """
        try:
            if specaxis is None:
                specaxis = self._determine_spectral_axis(header, data_shape)

            wavelength = self._extract_standard_wavelength(header, specaxis)
            if wavelength is None:
                wavelength = self._extract_special_cases(header, data_shape)

            return self._validate_wavelength(wavelength)
        except Exception as e:
            logger.warning(f"Error extrayendo wavelength: {e}")
            return None

    def _validate_wavelength(self, wavelength: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Valida que el array de wavelength sea consistente.

        Args:
            wavelength: Array a validar

        Returns:
            Array validado o None si es inválido
        """
        if wavelength is None:
            return None

        if not isinstance(wavelength, np.ndarray):
            try:
                wavelength = np.array(wavelength)
            except Exception:
                return None

        if len(wavelength) == 0:
            return None

        # Verificar que sea monótono creciente
        if not np.all(np.diff(wavelength) > 0):
            logger.warning("Array de wavelength no es monótono creciente")

        return wavelength

    def _determine_spectral_axis(self, header: pyfits.Header,
                                 data_shape: Optional[Tuple[int, ...]]) -> int:
        """
        Determina el eje espectral con validación mejorada.

        Args:
            header: Header FITS
            data_shape: Forma de los datos

        Returns:
            Número del eje espectral
        """
        dispaxis = header.get('DISPAXIS')
        if dispaxis is not None and isinstance(dispaxis, int) and dispaxis > 0:
            return dispaxis

        if data_shape and len(data_shape) == 3:
            return 3

        return 1

    def _extract_standard_wavelength(self, header: pyfits.Header,
                                     specaxis: int) -> Optional[np.ndarray]:
        """
        Extrae wavelength usando método estándar CRVAL/CDELT con validación.

        Args:
            header: Header FITS
            specaxis: Eje espectral

        Returns:
            Array de wavelengths o None
        """
        try:
            nwave = header.get(f'NAXIS{specaxis}')
            crval = header.get(f'CRVAL{specaxis}')
            cdelt = header.get(f'CDELT{specaxis}')

            if all(x is not None for x in [nwave, crval, cdelt]):
                if nwave <= 0 or cdelt == 0:
                    return None
                return crval + cdelt * np.arange(nwave)
        except Exception as e:
            logger.warning(f"Error en extracción estándar de wavelength: {e}")

        return None

    def _extract_special_cases(self, header: pyfits.Header,
                               data_shape: Optional[Tuple[int, ...]]) -> Optional[np.ndarray]:
        """
        Maneja casos especiales de instrumentos específicos con validación.

        Args:
            header: Header FITS
            data_shape: Forma de los datos

        Returns:
            Array de wavelengths o None
        """
        # MUSE
        if self._is_muse_case(header):
            return self._extract_muse_wavelength(header)

        # WEAVE RSS
        if self._is_weave_rss_case(header, data_shape):
            return self._extract_weave_wavelength(header)

        # Pycasso v2
        if self._is_pycasso_v2_case(header, data_shape):
            return self._extract_pycasso_wavelength(header, data_shape)

        return None

    def _is_muse_case(self, header: pyfits.Header) -> bool:
        """Verifica si es caso especial de MUSE con validación."""
        return ('CTYPE3' in header and
                isinstance(header.get('CTYPE3'), str) and
                'AWAV' in header['CTYPE3'] and
                'CD3_3' in header)

    def _extract_muse_wavelength(self, header: pyfits.Header) -> Optional[np.ndarray]:
        """
        Extrae wavelength para MUSE con validación.

        Args:
            header: Header FITS

        Returns:
            Array de wavelengths o None
        """
        try:
            crval = header.get('CRVAL3')
            cdelt = header.get('CD3_3')
            nwave = header.get('NAXIS3')

            if all(x is not None for x in [crval, cdelt, nwave]):
                if nwave <= 0 or cdelt == 0:
                    return None
                return crval + cdelt * np.arange(nwave)
        except Exception as e:
            logger.warning(f"Error extrayendo wavelength MUSE: {e}")

        return None

    def _is_weave_rss_case(self, header: pyfits.Header,
                           data_shape: Optional[Tuple[int, ...]]) -> bool:
        """Verifica si es caso WEAVE RSS con validación."""
        return ('CRVAL1' in header and 'CD1_1' in header and
                data_shape is not None and len(data_shape) == 2)

    def _extract_weave_wavelength(self, header: pyfits.Header) -> Optional[np.ndarray]:
        """
        Extrae wavelength para WEAVE con validación.

        Args:
            header: Header FITS

        Returns:
            Array de wavelengths o None
        """
        try:
            crval = header.get('CRVAL1')
            cdelt = header.get('CD1_1')
            nwave = header.get('NAXIS1')

            if all(x is not None for x in [crval, cdelt, nwave]):
                if nwave <= 0 or cdelt == 0:
                    return None
                return crval + cdelt * np.arange(nwave)
        except Exception as e:
            logger.warning(f"Error extrayendo wavelength WEAVE: {e}")

        return None

    def _is_pycasso_v2_case(self, header: pyfits.Header,
                            data_shape: Optional[Tuple[int, ...]]) -> bool:
        """Verifica si es caso Pycasso v2 con validación."""
        return ('PYCASSO VERSION' in header and data_shape is not None and
                len(data_shape) >= 1)

    def _extract_pycasso_wavelength(self, header: pyfits.Header,
                                    data_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """
        Extrae wavelength para Pycasso v2 con validación.

        Args:
            header: Header FITS
            data_shape: Forma de los datos

        Returns:
            Array de wavelengths o None
        """
        try:
            wcs = WCS(header)
            w = wcs.sub([3])
            pix_coords = np.arange(data_shape[0])
            wave_coords = w.wcs_pix2world(pix_coords[:, np.newaxis], 0)

            if w.wcs.cunit[0] == "m":
                wave_coords *= 1e10

            result = np.squeeze(wave_coords)
            return result if len(result) > 0 else None
        except Exception as e:
            logger.warning(f"Error extrayendo wavelength Pycasso v2: {e}")
            return None


# Resto de clases mantienen la misma estructura pero con mejor documentación y manejo de errores...

class PositionTableLoader:
    """
    Servicio especializado para cargar tablas de posiciones con validación mejorada.

    Maneja diferentes formatos de tablas de posiciones
    de diversos instrumentos astronómicos.
    """

    def __init__(self, validator: Optional[FileValidator] = None):
        """
        Inicializa el cargador de tablas de posiciones.

        Args:
            validator: Validador de archivos (opcional)
        """
        self._coordinate_processor = CoordinateProcessor()
        self._validator = validator or FileValidator()

    def load_position_table(self, filename: str,
                            filetype: str = "C",
                            angle: Optional[float] = None,
                            skycoord: bool = True) -> Dict[str, Any]:
        """
        Carga una tabla de posiciones desde archivo con validación.

        Args:
            filename: Ruta del archivo
            filetype: Tipo de fibra ('C' circular, 'H' hexagonal)
            angle: Ángulo de rotación en grados
            skycoord: Usar coordenadas de cielo

        Returns:
            Diccionario con información de posiciones validada

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato es inválido
            IOError: Si hay error cargando el archivo
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Archivo de posiciones "{filename}" no existe')

        try:
            if filename.endswith('.fits'):
                return self._load_fits_table(filename, angle, skycoord)
            else:
                return self._load_text_table(filename, angle)
        except Exception as e:
            logger.error(f"Error cargando tabla de posiciones {filename}: {e}")
            raise IOError(f'Error cargando tabla de posiciones: {e}')

    def _load_text_table(self, filename: str, angle: Optional[float]) -> Dict[str, Any]:
        """
        Carga tabla desde archivo de texto con validación.

        Args:
            filename: Ruta del archivo
            angle: Ángulo de rotación

        Returns:
            Diccionario con datos de posiciones
        """
        try:
            with open(filename, 'r') as f:
                header_line = f.readline().split()

                if len(header_line) < 3:
                    raise ValueError("Header de archivo inválido")

                filetype, xs, ys = header_line[0], float(header_line[1]), float(header_line[2])

            data = np.loadtxt(filename, skiprows=1, unpack=True)

            if len(data) < 3:
                raise ValueError("Archivo debe tener al menos 3 columnas")

            fiber_id, x, y = data[0], data[1], data[2]

            if angle is not None:
                x, y = self._coordinate_processor.rotate_coordinates(x, y, angle)

            return {
                'fiber_type': filetype,
                'fiber_size_x': xs,
                'fiber_size_y': ys,
                'id': fiber_id,
                'x': x,
                'y': y,
                'radius': xs if filetype == 'C' else None
            }
        except Exception as e:
            raise IOError(f"Error cargando archivo de texto: {e}")

    def _load_fits_table(self, filename: str, angle: Optional[float],
                         skycoord: bool) -> Dict[str, Any]:
        """
        Carga tabla desde archivo FITS con mejor manejo de errores.

        Args:
            filename: Ruta del archivo
            angle: Ángulo de rotación
            skycoord: Usar coordenadas de cielo

        Returns:
            Diccionario con datos de posiciones
        """
        loaders = [
            self._try_load_califa,
            self._try_load_megara,
            lambda f, a, s: self._try_load_lifu(f, a, s)
        ]

        last_error = None
        for loader in loaders:
            try:
                if loader == loaders[2]:  # LIFU loader needs skycoord parameter
                    return loader(filename, angle, skycoord)
                else:
                    return loader(filename, angle)
            except Exception as e:
                last_error = e
                continue

        raise IOError(f"No se pudo determinar el formato de la tabla: {filename}. "
                      f"Último error: {last_error}")

    def _try_load_califa(self, filename: str, angle: Optional[float]) -> Dict[str, Any]:
        """
        Intenta cargar como tabla CALIFA con validación.

        Args:
            filename: Ruta del archivo
            angle: Ángulo de rotación

        Returns:
            Diccionario con datos CALIFA
        """
        try:
            try:
                data, header = pyfits.getdata(filename, extname='FIBERS', header=True)
            except:
                data, header = pyfits.getdata(filename, ext=1, header=True)

            # Validar header requerido
            required_keys = ['FIBSHAPE', 'FIBSIZEX', 'FIBSIZEY', 'TTYPE1', 'TTYPE2']
            for key in required_keys:
                if key not in header:
                    raise ValueError(f"Clave requerida {key} no encontrada en header")

            filetype = header['FIBSHAPE'].strip()
            xs = header['FIBSIZEX']
            ys = header['FIBSIZEY']

            x = data[header['TTYPE1']]
            y = data[header['TTYPE2']]

            if angle is not None:
                x, y = self._coordinate_processor.rotate_coordinates(x, y, angle)

            return {
                'fiber_type': filetype,
                'fiber_size_x': xs,
                'fiber_size_y': ys,
                'x': x,
                'y': y,
                'radius': xs if filetype == 'C' else None
            }
        except Exception as e:
            raise ValueError(f"Error cargando tabla CALIFA: {e}")

    def _try_load_megara(self, filename: str, angle: Optional[float]) -> Dict[str, Any]:
        """
        Intenta cargar como tabla MEGARA con validación.

        Args:
            filename: Ruta del archivo
            angle: Ángulo de rotación

        Returns:
            Diccionario con datos MEGARA
        """
        try:
            header = pyfits.getheader(filename, extname='FIBERS')
            fibers = self._parse_megara_fibers(header)

            if not fibers:
                raise ValueError("No se encontraron fibras en header MEGARA")

            ids = list(fibers.keys())
            x = np.array([fibers[fid]['X'] for fid in ids])
            y = np.array([fibers[fid]['Y'] for fid in ids])

            if angle is not None:
                x, y = self._coordinate_processor.rotate_coordinates(x, y, angle)

            radius = self._coordinate_processor.calculate_fiber_radius(x, y)

            return {
                'fiber_type': 'H',  # MEGARA usa fibras hexagonales
                'id': np.array(ids),
                'x': x,
                'y': y,
                'radius': radius
            }
        except Exception as e:
            raise ValueError(f"Error cargando tabla MEGARA: {e}")

    def _parse_megara_fibers(self, header: pyfits.Header) -> Dict[int, Dict[str, Any]]:
        """
        Parsea información de fibras MEGARA del header con validación.

        Args:
            header: Header FITS

        Returns:
            Diccionario de fibras parseadas
        """
        fibers = {}

        for key in header:
            if key.startswith('FIB') and '_' in key:
                try:
                    fid = int(key.split('_')[0].replace('FIB', ''))
                    if fid not in fibers:
                        fibers[fid] = {}
                    param = key.split('_')[-1]
                    fibers[fid][param] = header[key]
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parseando clave MEGARA {key}: {e}")
                    continue

        return fibers

    def _try_load_lifu(self, filename: str, angle: Optional[float],
                       skycoord: bool) -> Dict[str, Any]:
        """
        Intenta cargar como tabla LIFU/WEAVE con validación.

        Args:
            filename: Ruta del archivo
            angle: Ángulo de rotación
            skycoord: Usar coordenadas de cielo

        Returns:
            Diccionario con datos LIFU
        """
        try:
            table = Table.read(filename, hdu='FIBTABLE')

            if skycoord:
                x, y = self._process_skycoord(table)
            else:
                # Validar columnas requeridas
                required_cols = ['XPOSITION', 'YPOSITION']
                for col in required_cols:
                    if col not in table.colnames:
                        raise ValueError(f"Columna requerida {col} no encontrada")

                x = table['XPOSITION']
                y = table['YPOSITION']

            if angle is not None:
                x, y = self._coordinate_processor.rotate_coordinates(x, y, angle)

            # Determinar tamaño de fibra
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
        except Exception as e:
            raise ValueError(f"Error cargando tabla LIFU: {e}")

    def _process_skycoord(self, table: Table) -> Tuple[np.ndarray, np.ndarray]:
        """
        Procesa coordenadas de cielo con validación.

        Args:
            table: Tabla con coordenadas

        Returns:
            Tupla (x, y) de coordenadas procesadas
        """
        try:
            # Validar columnas requeridas
            required_cols = ['FIBRERA', 'FIBREDEC', 'XPOSITION', 'YPOSITION']
            for col in required_cols:
                if col not in table.colnames:
                    raise ValueError(f"Columna requerida {col} no encontrada para skycoord")

            coords = SkyCoord(table['FIBRERA'] * u.deg, table['FIBREDEC'] * u.deg)
            ref_idx = np.argmin(table['XPOSITION'] ** 2 + table['YPOSITION'] ** 2)
            ra, dec = coords.spherical_offsets_to(coords[ref_idx])
            return ra.to(u.arcsec).value, dec.to(u.arcsec).value
        except Exception as e:
            logger.warning(f"Error procesando skycoord: {e}")
            return table['XPOSITION'], table['YPOSITION']


class CoordinateProcessor:
    """Procesador de coordenadas y transformaciones espaciales con validación."""

    def rotate_coordinates(self, x: np.ndarray, y: np.ndarray,
                           angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rota coordenadas por un ángulo dado con validación.

        Args:
            x: Coordenadas X
            y: Coordenadas Y
            angle: Ángulo de rotación en grados

        Returns:
            Tupla con coordenadas rotadas (x_rot, y_rot)

        Raises:
            ValueError: Si las coordenadas no tienen la misma forma
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if x.shape != y.shape:
            raise ValueError("Las coordenadas X e Y deben tener la misma forma")

        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        return x_rot, y_rot

    def calculate_fiber_radius(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula radio de fibra basado en distancias con validación.

        Args:
            x: Coordenadas X
            y: Coordenadas Y

        Returns:
            Radio calculado de la fibra
        """
        if len(x) != len(y):
            raise ValueError("Coordenadas X e Y deben tener la misma longitud")

        if len(x) < 2:
            return 1.0  # Valor por defecto

        try:
            from scipy.spatial.distance import cdist
            points = np.column_stack([x, y])
            distances = cdist(points, points, 'euclidean')
            distances[distances == 0] = np.inf
            min_distance = np.min(distances)
            return min_distance / 2.0
        except ImportError:
            logger.warning("scipy no disponible, usando radio por defecto")
            return 1.0
        except Exception as e:
            logger.warning(f"Error calculando radio de fibra: {e}")
            return 1.0


# Resto de clases (DataFactory, AstronomicalCalculator, DataProcessor, FileIOService)
# mantienen la misma implementación con documentación mejorada...

class DataFactory:
    """
    Factory para crear objetos de datos astronómicos con validación.

    Centraliza la creación de objetos SpectrumData, CubeData
    y otros tipos de datos astronómicos.
    """

    def create_spectrum_data(self, wavelength: np.ndarray, flux: np.ndarray,
                             error: Optional[np.ndarray] = None,
                             flag: Optional[np.ndarray] = None,
                             metadata: Optional[Dict] = None) -> 'SpectrumData':
        """
        Crea un objeto SpectrumData con validación.

        Args:
            wavelength: Array de longitudes de onda
            flux: Array de flujo
            error: Array de errores (opcional)
            flag: Array de flags (opcional)
            metadata: Metadatos adicionales

        Returns:
            Objeto SpectrumData creado

        Raises:
            ValueError: Si los arrays no tienen formas compatibles
        """
        # Validación básica
        if not isinstance(wavelength, np.ndarray):
            wavelength = np.array(wavelength)
        if not isinstance(flux, np.ndarray):
            flux = np.array(flux)

        if len(wavelength) != len(flux):
            raise ValueError("Wavelength y flux deben tener la misma longitud")

        if error is not None:
            if not isinstance(error, np.ndarray):
                error = np.array(error)
            if len(error) != len(flux):
                raise ValueError("Error debe tener la misma longitud que flux")

        if flag is not None:
            if not isinstance(flag, np.ndarray):
                flag = np.array(flag)
            if len(flag) != len(flux):
                raise ValueError("Flag debe tener la misma longitud que flux")

        from ..domain.models.spectrum_data import SpectrumData
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
                         metadata: Optional[Dict] = None) -> 'CubeData':
        """
        Crea un objeto CubeData con validación.

        Args:
            data: Array 3D de datos (lambda, y, x)
            wavelength: Array de longitudes de onda
            error: Array de errores
            flag: Array de flags
            metadata: Metadatos adicionales

        Returns:
            Objeto CubeData creado

        Raises:
            ValueError: Si los datos no tienen la forma correcta
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 3:
            raise ValueError("Los datos del cubo deben ser un array 3D")

        if wavelength is not None:
            if not isinstance(wavelength, np.ndarray):
                wavelength = np.array(wavelength)
            if len(wavelength) != data.shape[0]:
                raise ValueError("Wavelength debe tener la misma longitud que el eje espectral")

        if error is not None:
            if not isinstance(error, np.ndarray):
                error = np.array(error)
            if error.shape != data.shape:
                raise ValueError("Error debe tener la misma forma que los datos")

        if flag is not None:
            if not isinstance(flag, np.ndarray):
                flag = np.array(flag)
            if flag.shape != data.shape:
                raise ValueError("Flag debe tener la misma forma que los datos")

        from ..domain.models.cube_data import CubeData
        return CubeData(
            data=data,
            wavelength=wavelength,
            error=error,
            flag=flag,
            meta=metadata or {}
        )

    def create_astronomical_object(self, name: str,
                                   object_type: str = "GALAXY",
                                   redshift: Optional[float] = None,
                                   velocity: Optional[float] = None,
                                   coordinates: Optional['SpatialCoordinate'] = None) -> 'AstronomicalObject':
        """
        Crea un objeto astronómico con validación.

        Args:
            name: Nombre del objeto
            object_type: Tipo de objeto
            redshift: Corrimiento al rojo
            velocity: Velocidad radial
            coordinates: Coordenadas del objeto

        Returns:
            Objeto AstronomicalObject creado

        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not name or not isinstance(name, str):
            raise ValueError("El nombre debe ser una cadena no vacía")

        if redshift is not None and redshift < -1:
            raise ValueError("Redshift no puede ser menor a -1")

        if velocity is not None and abs(velocity) > 300000:  # km/s
            logger.warning(f"Velocidad muy alta: {velocity} km/s")

        from ..domain.entities.astronomical_entities import AstronomicalObject
        return AstronomicalObject(
            name=name,
            object_type=object_type,
            redshift=redshift,
            velocity=velocity,
            coordinates=coordinates
        )


class AstronomicalCalculator:
    """
    Calculadora para operaciones astronómicas comunes con validación.

    Centraliza cálculos relacionados con redshift, velocidades
    y correcciones astronómicas.
    """

    def __init__(self, speed_of_light: float = 299792.458):
        """
        Inicializa la calculadora astronómica.

        Args:
            speed_of_light: Velocidad de la luz en km/s

        Raises:
            ValueError: Si la velocidad de la luz es inválida
        """
        if speed_of_light <= 0:
            raise ValueError("La velocidad de la luz debe ser positiva")
        self.speed_of_light = speed_of_light

    def apply_velocity_correction(self, wavelength: np.ndarray,
                                  velocity: float) -> np.ndarray:
        """
        Aplica corrección de velocidad a longitudes de onda con validación.

        Args:
            wavelength: Array de longitudes de onda
            velocity: Velocidad en km/s

        Returns:
            Array de longitudes de onda corregidas

        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not isinstance(wavelength, np.ndarray):
            wavelength = np.array(wavelength)

        if abs(velocity) > self.speed_of_light:
            raise ValueError(f"Velocidad {velocity} km/s excede la velocidad de la luz")

        z = velocity / self.speed_of_light
        if z <= -1:
            raise ValueError(f"Redshift calculado {z} es inválido (≤ -1)")

        return wavelength / (1.0 + z)

    def calculate_redshift_from_velocity(self, velocity: float) -> float:
        """
        Calcula redshift desde velocidad con validación.

        Args:
            velocity: Velocidad en km/s

        Returns:
            Redshift calculado

        Raises:
            ValueError: Si la velocidad es inválida
        """
        if abs(velocity) > self.speed_of_light:
            raise ValueError(f"Velocidad {velocity} km/s excede la velocidad de la luz")

        return velocity / self.speed_of_light

    def calculate_velocity_from_redshift(self, redshift: float) -> float:
        """
        Calcula velocidad desde redshift con validación.

        Args:
            redshift: Corrimiento al rojo

        Returns:
            Velocidad en km/s

        Raises:
            ValueError: Si el redshift es inválido
        """
        if redshift <= -1:
            raise ValueError(f"Redshift {redshift} es inválido (≤ -1)")

        velocity = redshift * self.speed_of_light
        if abs(velocity) > self.speed_of_light:
            logger.warning(f"Velocidad calculada {velocity} km/s excede c")

        return velocity


class DataProcessor:
    """
    Procesador de datos astronómicos con validación mejorada.

    Proporciona operaciones de procesamiento y análisis
    de datos espectrales y espaciales.
    """

    def get_flux_limits(self, data: np.ndarray,
                        percentile: float = 0.1) -> Tuple[float, float]:
        """
        Calcula límites de flujo para visualización con validación.

        Args:
            data: Array de datos
            percentile: Percentil para calcular límites

        Returns:
            Tupla (min, max) de límites de flujo

        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if not (0 < percentile < 50):
            raise ValueError("Percentil debe estar entre 0 y 50")

        valid_data = data[np.isfinite(data)]
        if valid_data.size == 0:
            logger.warning("No hay datos válidos para calcular límites")
            return (np.nan, np.nan)

        try:
            p_low = np.percentile(valid_data, percentile)
            p_high = np.percentile(valid_data, 100 - percentile)
            return (float(p_low), float(p_high))
        except Exception as e:
            logger.error(f"Error calculando límites de flujo: {e}")
            return (np.nan, np.nan)

    def mask_flagged_data(self, data: np.ndarray, flags: np.ndarray,
                          flag_value: int = 0) -> np.ndarray:
        """
        Aplica máscara de flags a los datos con validación.

        Args:
            data: Array de datos
            flags: Array de flags
            flag_value: Valor de flag que indica datos válidos

        Returns:
            Array enmascarado

        Raises:
            ValueError: Si los arrays no son compatibles
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if not isinstance(flags, np.ndarray):
            flags = np.array(flags)

        if data.shape != flags.shape:
            raise ValueError("Data y flags deben tener la misma forma")

        mask = flags > flag_value
        return np.ma.array(data, mask=mask)

    def extract_spectrum_from_cube(self, cube_data: 'CubeData',
                                   x: int, y: int) -> Optional['SpectrumData']:
        """
        Extrae un espectro de un spaxel específico del cubo con validación.

        Args:
            cube_data: Objeto CubeData
            x: Coordenada X del spaxel
            y: Coordenada Y del spaxel

        Returns:
            Objeto SpectrumData del spaxel o None si las coordenadas son inválidas

        Raises:
            ValueError: Si las coordenadas están fuera de rango
        """
        if not hasattr(cube_data, 'get_spaxel_spectrum'):
            raise AttributeError("CubeData debe tener método get_spaxel_spectrum")

        if not (0 <= x < cube_data.n_x):
            raise ValueError(f"Coordenada X {x} fuera de rango [0, {cube_data.n_x})")
        if not (0 <= y < cube_data.n_y):
            raise ValueError(f"Coordenada Y {y} fuera de rango [0, {cube_data.n_y})")

        return cube_data.get_spaxel_spectrum(x, y)

    def calculate_integrated_spectrum(self, cube_data: 'CubeData',
                                      mask: Optional[np.ndarray] = None) -> 'SpectrumData':
        """
        Calcula el espectro integrado de un cubo con validación.

        Args:
            cube_data: Objeto CubeData
            mask: Máscara opcional para seleccionar spaxels

        Returns:
            Espectro integrado como SpectrumData

        Raises:
            ValueError: Si la máscara no es compatible
        """
        if not hasattr(cube_data, 'get_mean_spectrum'):
            raise AttributeError("CubeData debe tener método get_mean_spectrum")

        if mask is not None:
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            if mask.shape != (cube_data.n_y, cube_data.n_x):
                raise ValueError("Máscara debe tener forma (n_y, n_x)")

        return cube_data.get_mean_spectrum(mask)


class FileIOService:
    """
    Servicio de entrada/salida de archivos con validación mejorada.

    Maneja operaciones de guardado de espectros
    y otros datos astronómicos.
    """

    def save_spectrum(self, spectrum: 'SpectrumData', filename: str,
                      format: str = 'txt',
                      header: Optional[pyfits.Header] = None) -> None:
        """
        Guarda un espectro a archivo con validación.

        Args:
            spectrum: Objeto SpectrumData
            filename: Nombre del archivo
            format: Formato ('txt' o 'fits')
            header: Header FITS opcional

        Raises:
            ValueError: Si el formato no es soportado o los datos son inválidos
            IOError: Si hay error escribiendo el archivo
        """
        if not hasattr(spectrum, 'wavelength') or not hasattr(spectrum, 'flux'):
            raise ValueError("Spectrum debe tener atributos wavelength y flux")

        if len(spectrum.wavelength) != len(spectrum.flux):
            raise ValueError("Wavelength y flux deben tener la misma longitud")

        if format.lower() == 'txt':
            self._save_spectrum_txt(spectrum, filename)
        elif format.lower() == 'fits':
            self._save_spectrum_fits(spectrum, filename, header)
        else:
            raise ValueError(f"Formato no soportado: {format}")

    def _save_spectrum_txt(self, spectrum: 'SpectrumData', filename: str) -> None:
        """
        Guarda espectro en formato texto con validación.

        Args:
            spectrum: Objeto SpectrumData
            filename: Nombre del archivo

        Raises:
            IOError: Si hay error escribiendo el archivo
        """
        try:
            data = np.column_stack([spectrum.wavelength, spectrum.flux])

            if spectrum.error is not None and len(spectrum.error) == len(spectrum.flux):
                data = np.column_stack([data, spectrum.error])

            # Crear header informativo
            header_lines = [
                f"# Spectrum data saved at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"# Wavelength range: {np.min(spectrum.wavelength):.2f} - {np.max(spectrum.wavelength):.2f}",
                f"# Flux range: {np.min(spectrum.flux):.6e} - {np.max(spectrum.flux):.6e}",
                f"# Number of points: {len(spectrum.wavelength)}"
            ]

            # Añadir metadatos si existen
            if hasattr(spectrum, 'meta') and spectrum.meta:
                header_lines.append(f"# Metadata: {spectrum.meta}")

            np.savetxt(filename, data, header='\n'.join(header_lines), fmt='%.6e')
            logger.info(f"Espectro guardado en formato texto: {filename}")

        except Exception as e:
            raise IOError(f"Error guardando espectro en formato texto: {e}")

    def _save_spectrum_fits(self, spectrum: 'SpectrumData', filename: str,
                            header: Optional[pyfits.Header]) -> None:
        """
        Guarda espectro en formato FITS con validación.

        Args:
            spectrum: Objeto SpectrumData
            filename: Nombre del archivo
            header: Header FITS opcional

        Raises:
            IOError: Si hay error escribiendo el archivo
        """
        try:
            # Preparar datos
            if spectrum.error is not None and len(spectrum.error) == len(spectrum.flux):
                data = np.column_stack([spectrum.flux, spectrum.error])
            else:
                data = spectrum.flux

            # Crear HDU
            hdu = pyfits.PrimaryHDU(data)

            # Añadir header existente si se proporciona
            if header is not None:
                hdu.header.update(header)

            # Añadir información de wavelength si es posible
            if len(spectrum.wavelength) > 1:
                wave_diff = np.diff(spectrum.wavelength)
                if np.allclose(wave_diff, wave_diff[0], rtol=1e-6):
                    # Wavelength spacing constante
                    crval = float(spectrum.wavelength[0])
                    cdelt = float(wave_diff[0])
                    hdu.header['CRVAL1'] = crval
                    hdu.header['CDELT1'] = cdelt
                    hdu.header['CRPIX1'] = 1
                    hdu.header['CTYPE1'] = 'WAVELENGTH'

            # Añadir metadatos del espectro
            hdu.header['COMMENT'] = f'Spectrum saved at {time.strftime("%Y-%m-%d %H:%M:%S")}'
            hdu.header['NPOINTS'] = len(spectrum.wavelength)

            # Añadir metadatos personalizados si existen
            if hasattr(spectrum, 'meta') and spectrum.meta:
                for key, value in spectrum.meta.items():
                    if isinstance(value, (str, int, float, bool)):
                        try:
                            hdu.header[f'META_{key.upper()}'[:8]] = value
                        except Exception as e:
                            logger.warning(f"No se pudo añadir metadato {key}: {e}")

            hdu.writeto(filename, overwrite=True)
            logger.info(f"Espectro guardado en formato FITS: {filename}")

        except Exception as e:
            raise IOError(f"Error guardando espectro en formato FITS: {e}")


class DataService:
    """
    Servicio principal de datos refactorizado con validación mejorada.

    Actúa como fachada que coordina los diferentes servicios especializados,
    manteniendo la misma interfaz externa que la versión original pero con
    una arquitectura interna modular y de baja complejidad.
    """

    def __init__(self, validator: Optional[FileValidator] = None):
        """
        Inicializa el servicio de datos con validación mejorada.

        Aplica el patrón de inyección de dependencias para facilitar
        las pruebas y el mantenimiento del código.

        Args:
            validator: Validador de archivos personalizado (opcional)
        """
        self.speed_of_light = 299792.458  # km/s
        self._loaded_files = {}
        self._cache = {}

        # Inyección de dependencias para servicios especializados
        validator = validator or FileValidator()
        self._fits_loader = FitsFileLoader(validator)
        self._position_loader = PositionTableLoader(validator)
        self._data_factory = DataFactory()
        self._calculator = AstronomicalCalculator(self.speed_of_light)
        self._processor = DataProcessor()
        self._file_io = FileIOService()

        logger.info("DataService inicializado con arquitectura modular mejorada")

    # === MÉTODOS PÚBLICOS DE LA INTERFAZ ORIGINAL ===

    def load_fits_file(self, filename: str, **kwargs) -> Dict[str, Any]:
        """
        Carga un archivo FITS manteniendo la interfaz original con validación.

        Delega la operación al servicio especializado FitsFileLoader
        para reducir la complejidad de este método.

        Args:
            filename: Ruta al archivo FITS
            **kwargs: Parámetros de configuración

        Returns:
            Diccionario con datos cargados (compatible con versión original)

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el archivo es inválido
            IOError: Si hay error cargando el archivo
        """
        try:
            result = self._fits_loader.load_fits_file(filename, **kwargs)
            converted_result = self._convert_load_result_to_dict(result)

            # Guardar en caché para referencia futura
            self._loaded_files[filename] = converted_result

            return converted_result
        except Exception as e:
            logger.error(f"Error en load_fits_file para {filename}: {e}")
            raise

    def create_spectrum_data(self, wavelength: np.ndarray, flux: np.ndarray,
                             error: Optional[np.ndarray] = None,
                             flag: Optional[np.ndarray] = None,
                             metadata: Optional[Dict] = None) -> 'SpectrumData':
        """
        Crea un objeto SpectrumData con validación.

        Delega al factory especializado manteniendo la misma interfaz.

        Args:
            wavelength: Array de longitudes de onda
            flux: Array de flujo
            error: Array de errores (opcional)
            flag: Array de flags (opcional)
            metadata: Metadatos adicionales

        Returns:
            Objeto SpectrumData validado
        """
        return self._data_factory.create_spectrum_data(
            wavelength, flux, error, flag, metadata
        )

    def create_cube_data(self, data: np.ndarray, **kwargs) -> 'CubeData':
        """
        Crea un objeto CubeData con validación.

        Delega al factory especializado manteniendo la misma interfaz.

        Args:
            data: Array 3D de datos
            **kwargs: Parámetros adicionales

        Returns:
            Objeto CubeData validado
        """
        return self._data_factory.create_cube_data(data, **kwargs)

    def extract_spectrum_from_cube(self, cube_data: 'CubeData',
                                   x: int, y: int) -> Optional['SpectrumData']:
        """
        Extrae espectro de un cubo con validación.

        Delega al procesador especializado.

        Args:
            cube_data: Objeto CubeData
            x: Coordenada X del spaxel
            y: Coordenada Y del spaxel

        Returns:
            Objeto SpectrumData del spaxel
        """
        return self._processor.extract_spectrum_from_cube(cube_data, x, y)

    def calculate_integrated_spectrum(self, cube_data: 'CubeData',
                                      mask: Optional[np.ndarray] = None) -> 'SpectrumData':
        """
        Calcula espectro integrado con validación.

        Delega al procesador especializado.

        Args:
            cube_data: Objeto CubeData
            mask: Máscara opcional para seleccionar spaxels

        Returns:
            Espectro integrado validado
        """
        return self._processor.calculate_integrated_spectrum(cube_data, mask)

    def load_position_table(self, filename: str, **kwargs) -> Dict[str, Any]:
        """
        Carga tabla de posiciones con validación.

        Delega al cargador especializado manteniendo la interfaz original.

        Args:
            filename: Ruta del archivo
            **kwargs: Parámetros adicionales

        Returns:
            Diccionario con información de posiciones validada
        """
        return self._position_loader.load_position_table(filename, **kwargs)

    def create_astronomical_object(self, name: str, **kwargs) -> 'AstronomicalObject':
        """
        Crea objeto astronómico con validación.

        Delega al factory especializado.

        Args:
            name: Nombre del objeto
            **kwargs: Parámetros adicionales

        Returns:
            Objeto astronómico validado
        """
        return self._data_factory.create_astronomical_object(name, **kwargs)

    def apply_velocity_correction(self, wavelength: np.ndarray,
                                  velocity: float) -> np.ndarray:
        """
        Aplica corrección de velocidad con validación.

        Delega al calculador especializado.

        Args:
            wavelength: Array de longitudes de onda
            velocity: Velocidad en km/s

        Returns:
            Array de longitudes de onda corregidas
        """
        return self._calculator.apply_velocity_correction(wavelength, velocity)

    def calculate_redshift_from_velocity(self, velocity: float) -> float:
        """
        Calcula redshift desde velocidad con validación.

        Delega al calculador especializado.

        Args:
            velocity: Velocidad en km/s

        Returns:
            Redshift calculado
        """
        return self._calculator.calculate_redshift_from_velocity(velocity)

    def calculate_velocity_from_redshift(self, redshift: float) -> float:
        """
        Calcula velocidad desde redshift con validación.

        Delega al calculador especializado.

        Args:
            redshift: Corrimiento al rojo

        Returns:
            Velocidad calculada en km/s
        """
        return self._calculator.calculate_velocity_from_redshift(redshift)

    def get_flux_limits(self, data: np.ndarray, **kwargs) -> Tuple[float, float]:
        """
        Calcula límites de flujo con validación.

        Delega al procesador especializado.

        Args:
            data: Array de datos
            **kwargs: Parámetros adicionales

        Returns:
            Tupla (min, max) de límites de flujo
        """
        return self._processor.get_flux_limits(data, **kwargs)

    def mask_flagged_data(self, data: np.ndarray, flags: np.ndarray,
                          flag_value: int = 0) -> np.ndarray:
        """
        Aplica máscara de flags con validación.

        Delega al procesador especializado.

        Args:
            data: Array de datos
            flags: Array de flags
            flag_value: Valor de flag para datos válidos

        Returns:
            Array enmascarado
        """
        return self._processor.mask_flagged_data(data, flags, flag_value)

    def save_spectrum(self, spectrum: 'SpectrumData', filename: str,
                      format: str = 'txt', **kwargs) -> None:
        """
        Guarda espectro a archivo con validación.

        Delega al servicio de E/S especializado.

        Args:
            spectrum: Objeto SpectrumData
            filename: Nombre del archivo
            format: Formato de archivo
            **kwargs: Parámetros adicionales
        """
        self._file_io.save_spectrum(spectrum, filename, format, **kwargs)

    # === MÉTODOS ADICIONALES PARA GESTIÓN MEJORADA ===

    def clear_cache(self) -> None:
        """Limpia la caché de archivos cargados."""
        self._loaded_files.clear()
        self._cache.clear()
        logger.info("Caché del DataService limpiada")

    def get_loaded_files(self) -> List[str]:
        """
        Obtiene la lista de archivos cargados.

        Returns:
            Lista de nombres de archivos cargados
        """
        return list(self._loaded_files.keys())

    def validate_fits_file(self, filename: str) -> bool:
        """
        Valida un archivo FITS sin cargarlo completamente.

        Args:
            filename: Ruta al archivo FITS

        Returns:
            True si el archivo es válido
        """
        try:
            return self._fits_loader._validator.validate_fits_file(filename)
        except Exception:
            return False

    def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información básica de un archivo cargado.

        Args:
            filename: Nombre del archivo

        Returns:
            Diccionario con información del archivo o None
        """
        return self._loaded_files.get(filename)

    # === MÉTODOS PRIVADOS DE CONVERSIÓN ===

    def _convert_load_result_to_dict(self, result: LoadResult) -> Dict[str, Any]:
        """
        Convierte LoadResult al formato diccionario de la versión original.

        Mantiene compatibilidad total con el código existente.

        Args:
            result: Resultado de carga a convertir

        Returns:
            Diccionario en formato compatible
        """
        return {
            'filename': result.metadata.get('filename', ''),
            'data': result.data,
            'header': result.header,
            'primary_header': result.primary_header,
            'wavelength': result.wavelength,
            'error': result.error,
            'flag': result.flag,
            'wave_rest': result.metadata.get('wave_rest'),
            'velocity': result.metadata.get('velocity'),
            'redshift': result.metadata.get('redshift'),
            'instrument': result.metadata.get('instrument'),
            'survey': result.metadata.get('survey'),
            'califaid': result.metadata.get('califaid'),
            'pycasso_version': result.metadata.get('pycasso_version'),
            'load_timestamp': result.metadata.get('load_timestamp'),
            'metadata': result.metadata
        }
