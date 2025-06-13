"""
Módulo para el manejo de datos de cubos espectrales 3D.

Este módulo proporciona las clases y funciones necesarias para trabajar
con datos de cubos espectrales astronómicos siguiendo el principio de
responsabilidad única y buenas prácticas de programación.

Versión: 2.0.0
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from abc import ABC, abstractmethod
import logging


class SpectrumData:
    """
    Contenedor para datos de un espectro individual.

    Esta clase encapsula los datos de un espectro unidimensional
    incluyendo flujo, error, flags y longitudes de onda asociadas.
    """

    def __init__(self, wavelength: Union[np.ndarray, list],
                 flux: Union[np.ndarray, list],
                 error: Optional[Union[np.ndarray, list]] = None,
                 flag: Optional[Union[np.ndarray, list]] = None,
                 meta: Optional[Dict[str, Any]] = None):
        """
        Inicializa un objeto SpectrumData.

        Args:
            wavelength: Longitudes de onda del espectro
            flux: Valores de flujo del espectro
            error: Errores asociados al flujo (opcional)
            flag: Flags de calidad (opcional)
            meta: Metadatos adicionales (opcional)
        """
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.error = np.asarray(error, dtype=np.float64) if error is not None else None
        self.flag = np.asarray(flag, dtype=np.float64) if flag is not None else None
        self.meta = meta if meta is not None else {}


class DataValidator:
    """
    Validador de datos para cubos espectrales.

    Esta clase se encarga exclusivamente de validar que los datos
    cumplan con los requisitos necesarios para formar un cubo espectral válido.
    """

    @staticmethod
    def validate_cube_shape(data: np.ndarray) -> bool:
        """
        Valida que los datos tengan la forma correcta para un cubo 3D.

        Args:
            data: Array numpy a validar

        Returns:
            bool: True si la forma es válida, False en caso contrario

        Raises:
            ValueError: Si los datos no tienen la dimensionalidad correcta
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Los datos deben ser un array numpy")

        if data.ndim != 3:
            raise ValueError(f"Los datos deben ser 3D, recibido {data.ndim}D")

        if any(dim <= 0 for dim in data.shape):
            raise ValueError("Todas las dimensiones deben ser positivas")

        return True

    @staticmethod
    def validate_wavelength_consistency(data: np.ndarray, wavelength: Optional[np.ndarray]) -> bool:
        """
        Valida que las longitudes de onda sean consistentes con los datos.

        Args:
            data: Datos del cubo espectral
            wavelength: Array de longitudes de onda

        Returns:
            bool: True si son consistentes

        Raises:
            ValueError: Si hay inconsistencias
        """
        if wavelength is not None:
            if wavelength.size != data.shape[0]:
                raise ValueError(
                    f"El tamaño de wavelength ({wavelength.size}) debe coincidir "
                    f"con la primera dimensión de data ({data.shape[0]})"
                )
        return True

    @staticmethod
    def validate_auxiliary_data_shape(main_data: np.ndarray, aux_data: Optional[np.ndarray],
                                      data_name: str) -> bool:
        """
        Valida que los datos auxiliares (error, flag) tengan la forma correcta.

        Args:
            main_data: Datos principales del cubo
            aux_data: Datos auxiliares a validar
            data_name: Nombre de los datos auxiliares para el mensaje de error

        Returns:
            bool: True si la forma es válida

        Raises:
            ValueError: Si las formas no coinciden
        """
        if aux_data is not None:
            if aux_data.shape != main_data.shape:
                raise ValueError(
                    f"La forma de {data_name} {aux_data.shape} debe coincidir "
                    f"con la forma de data {main_data.shape}"
                )
        return True


class CoordinateManager:
    """
    Gestor de coordenadas para cubos espectrales.

    Maneja la conversión y validación de coordenadas espaciales y espectrales.
    """

    def __init__(self, n_lambda: int, n_y: int, n_x: int):
        """
        Inicializa el gestor de coordenadas.

        Args:
            n_lambda: Número de canales espectrales
            n_y: Número de píxeles en dirección Y
            n_x: Número de píxeles en dirección X
        """
        self.n_lambda = n_lambda
        self.n_y = n_y
        self.n_x = n_x

    def validate_spatial_coordinates(self, x: int, y: int) -> bool:
        """
        Valida que las coordenadas espaciales estén dentro de los límites.

        Args:
            x: Coordenada X
            y: Coordenada Y

        Returns:
            bool: True si las coordenadas son válidas
        """
        return (0 <= x < self.n_x) and (0 <= y < self.n_y)

    def get_valid_coordinate_ranges(self) -> Dict[str, Tuple[int, int]]:
        """
        Obtiene los rangos válidos de coordenadas.

        Returns:
            Dict con los rangos válidos para cada dimensión
        """
        return {
            'x': (0, self.n_x - 1),
            'y': (0, self.n_y - 1),
            'lambda': (0, self.n_lambda - 1)
        }


class SpectrumExtractor:
    """
    Extractor de espectros del cubo de datos.

    Se encarga de extraer espectros individuales y realizar operaciones
    de agregación espacial.
    """

    def __init__(self, data: np.ndarray, wavelength: np.ndarray,
                 error: Optional[np.ndarray] = None, flag: Optional[np.ndarray] = None):
        """
        Inicializa el extractor de espectros.

        Args:
            data: Datos del cubo espectral (lambda, y, x)
            wavelength: Longitudes de onda
            error: Datos de error (opcional)
            flag: Datos de flags (opcional)
        """
        self.data = data
        self.wavelength = wavelength
        self.error = error
        self.flag = flag

    def extract_spaxel_spectrum(self, x: int, y: int) -> SpectrumData:
        """
        Extrae el espectro de un spaxel específico.

        Args:
            x: Coordenada X del spaxel
            y: Coordenada Y del spaxel

        Returns:
            SpectrumData: Objeto con el espectro extraído

        Raises:
            IndexError: Si las coordenadas están fuera de rango
        """
        try:
            flux = self.data[:, y, x]
            error = self.error[:, y, x] if self.error is not None else None
            flag = self.flag[:, y, x] if self.flag is not None else None

            return SpectrumData(self.wavelength, flux, error, flag)

        except IndexError as e:
            raise IndexError(f"Coordenadas fuera de rango: x={x}, y={y}") from e

    def extract_mean_spectrum(self, mask: Optional[np.ndarray] = None) -> SpectrumData:
        """
        Extrae el espectro promedio del cubo.

        Args:
            mask: Máscara booleana 2D para seleccionar spaxels (opcional)

        Returns:
            SpectrumData: Espectro promedio
        """
        if mask is not None:
            masked_data = self._apply_spatial_mask(mask)
            mean_flux = np.nanmean(masked_data, axis=(1, 2))
        else:
            mean_flux = np.nanmean(self.data, axis=(1, 2))

        # Calcular error promedio si existe
        mean_error = None
        if self.error is not None:
            if mask is not None:
                masked_error = self._apply_spatial_mask(mask, self.error)
                mean_error = np.nanmean(masked_error, axis=(1, 2))
            else:
                mean_error = np.nanmean(self.error, axis=(1, 2))

        # Para flags, usar promedio (podría cambiarse a OR lógico si se prefiere)
        mean_flag = None
        if self.flag is not None:
            if mask is not None:
                masked_flag = self._apply_spatial_mask(mask, self.flag)
                mean_flag = np.nanmean(masked_flag, axis=(1, 2))
            else:
                mean_flag = np.nanmean(self.flag, axis=(1, 2))

        return SpectrumData(self.wavelength, mean_flux, mean_error, mean_flag)

    def _apply_spatial_mask(self, mask: np.ndarray, data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Aplica una máscara espacial a los datos.

        Args:
            mask: Máscara booleana 2D
            data: Datos a enmascarar (usa self.data si es None)

        Returns:
            np.ndarray: Datos enmascarados
        """
        target_data = data if data is not None else self.data
        mask = np.asarray(mask)
        return np.where(mask, target_data, np.nan)


class StatisticsCalculator:
    """
    Calculadora de estadísticas para cubos espectrales.

    Proporciona métodos para calcular estadísticas globales y locales
    de los datos del cubo.
    """

    def __init__(self, data: np.ndarray):
        """
        Inicializa la calculadora de estadísticas.

        Args:
            data: Datos del cubo espectral
        """
        self.data = data

    def calculate_flux_limits(self) -> Tuple[float, float]:
        """
        Calcula los límites globales de flujo en el cubo.

        Returns:
            Tuple[float, float]: (mínimo, máximo) del flujo
        """
        valid_data = self.data[np.isfinite(self.data)]

        if valid_data.size == 0:
            logging.warning("No hay datos válidos para calcular límites de flujo")
            return (np.nan, np.nan)

        return (float(np.min(valid_data)), float(np.max(valid_data)))

    def calculate_statistics_summary(self) -> Dict[str, float]:
        """
        Calcula un resumen estadístico completo del cubo.

        Returns:
            Dict: Diccionario con estadísticas principales
        """
        valid_data = self.data[np.isfinite(self.data)]

        if valid_data.size == 0:
            return {
                'min': np.nan, 'max': np.nan, 'mean': np.nan,
                'median': np.nan, 'std': np.nan, 'valid_pixels': 0
            }

        return {
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'median': float(np.median(valid_data)),
            'std': float(np.std(valid_data)),
            'valid_pixels': int(valid_data.size)
        }


class CubeData:
    """
    Modelo de datos para un cubo espectral 3D astronómico.

    Esta clase encapsula los datos de un cubo espectral con dimensiones
    (lambda, y, x) y proporciona una interfaz para acceder y manipular
    estos datos de manera segura y eficiente.

    La clase sigue el principio de responsabilidad única: se encarga
    únicamente de mantener los datos del cubo y proporcionar acceso
    controlado a los mismos, delegando operaciones específicas a
    clases especializadas.

    Attributes:
        data (np.ndarray): Datos principales del cubo (lambda, y, x)
        wavelength (np.ndarray): Longitudes de onda correspondientes
        error (Optional[np.ndarray]): Datos de error asociados
        flag (Optional[np.ndarray]): Flags de calidad de datos
        meta (Dict[str, Any]): Metadatos adicionales
        n_lambda (int): Número de canales espectrales
        n_y (int): Número de píxeles en dirección Y
        n_x (int): Número de píxeles en dirección X

    Example:
        >>> import numpy as np
        >>> data = np.random.rand(100, 50, 50)  # lambda, y, x
        >>> wavelength = np.linspace(4000, 7000, 100)
        >>> cube = CubeData(data, wavelength=wavelength)
        >>> spectrum = cube.get_spaxel_spectrum(25, 25)
        >>> limits = cube.get_flux_limits()
    """

    def __init__(self, data: Union[np.ndarray, list],
                 wavelength: Optional[Union[np.ndarray, list]] = None,
                 error: Optional[Union[np.ndarray, list]] = None,
                 flag: Optional[Union[np.ndarray, list]] = None,
                 meta: Optional[Dict[str, Any]] = None):
        """
        Inicializa un nuevo cubo de datos espectrales.

        Args:
            data: Datos principales del cubo con forma (lambda, y, x)
            wavelength: Longitudes de onda. Si es None, se genera un array
                       con índices consecutivos
            error: Datos de error con la misma forma que data (opcional)
            flag: Flags de calidad con la misma forma que data (opcional)
            meta: Diccionario con metadatos adicionales (opcional)

        Raises:
            TypeError: Si los datos no son del tipo correcto
            ValueError: Si las dimensiones no son consistentes

        Example:
            >>> data = np.random.rand(10, 20, 20)
            >>> cube = CubeData(data)
            >>> print(cube.shape)  # (10, 20, 20)
        """
        # Convertir a arrays numpy y validar
        self.data = np.asarray(data, dtype=np.float64)

        # Validar estructura básica del cubo
        DataValidator.validate_cube_shape(self.data)

        # Procesar wavelength
        if wavelength is not None:
            self.wavelength = np.asarray(wavelength, dtype=np.float64)
            DataValidator.validate_wavelength_consistency(self.data, self.wavelength)
        else:
            self.wavelength = np.arange(self.data.shape[0], dtype=np.float64)

        # Procesar datos auxiliares
        self.error = self._process_auxiliary_data(error, "error")
        self.flag = self._process_auxiliary_data(flag, "flag")

        # Metadatos
        self.meta = meta if meta is not None else {}

        # Dimensiones
        self.n_lambda, self.n_y, self.n_x = self.data.shape

        # Inicializar componentes especializados
        self._coordinate_manager = CoordinateManager(self.n_lambda, self.n_y, self.n_x)
        self._spectrum_extractor = SpectrumExtractor(
            self.data, self.wavelength, self.error, self.flag
        )
        self._statistics_calculator = StatisticsCalculator(self.data)

        # Configurar logging
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"CubeData inicializado con forma {self.data.shape}")

    def _process_auxiliary_data(self, aux_data: Optional[Union[np.ndarray, list]],
                                data_name: str) -> Optional[np.ndarray]:
        """
        Procesa datos auxiliares (error, flag) validando su consistencia.

        Args:
            aux_data: Datos auxiliares a procesar
            data_name: Nombre de los datos para mensajes de error

        Returns:
            Optional[np.ndarray]: Datos procesados o None
        """
        if aux_data is not None:
            processed = np.asarray(aux_data, dtype=np.float64)
            DataValidator.validate_auxiliary_data_shape(self.data, processed, data_name)
            return processed
        return None

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Forma del cubo de datos.

        Returns:
            Tuple[int, int, int]: (n_lambda, n_y, n_x)
        """
        return self.data.shape

    @property
    def size(self) -> int:
        """
        Número total de elementos en el cubo.

        Returns:
            int: Número total de elementos
        """
        return self.data.size

    def get_spaxel_spectrum(self, x: int, y: int) -> Optional[SpectrumData]:
        """
        Extrae el espectro de un spaxel específico.

        Args:
            x: Coordenada X del spaxel (0 <= x < n_x)
            y: Coordenada Y del spaxel (0 <= y < n_y)

        Returns:
            SpectrumData: Objeto con el espectro extraído, o None si
                         las coordenadas están fuera de rango

        Example:
            >>> cube = CubeData(np.random.rand(10, 20, 20))
            >>> spectrum = cube.get_spaxel_spectrum(10, 10)
            >>> if spectrum is not None:
            ...     print(f"Flujo máximo: {spectrum.flux.max()}")
        """
        if not self._coordinate_manager.validate_spatial_coordinates(x, y):
            self._logger.warning(f"Coordenadas fuera de rango: x={x}, y={y}")
            return None

        try:
            return self._spectrum_extractor.extract_spaxel_spectrum(x, y)
        except Exception as e:
            self._logger.error(f"Error extrayendo espectro en ({x}, {y}): {e}")
            return None

    def get_mean_spectrum(self, mask: Optional[np.ndarray] = None) -> SpectrumData:
        """
        Calcula el espectro promedio del cubo o de una región enmascarada.

        Args:
            mask: Máscara booleana 2D para seleccionar spaxels (opcional).
                 Si es None, se promedian todos los spaxels.

        Returns:
            SpectrumData: Espectro promedio

        Raises:
            ValueError: Si la máscara no tiene la forma correcta

        Example:
            >>> cube = CubeData(np.random.rand(10, 20, 20))
            >>> # Espectro promedio de todo el cubo
            >>> mean_all = cube.get_mean_spectrum()
            >>> # Espectro promedio de una región central
            >>> mask = np.zeros((20, 20), dtype=bool)
            >>> mask[8:12, 8:12] = True
            >>> mean_center = cube.get_mean_spectrum(mask)
        """
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self.n_y, self.n_x):
                raise ValueError(
                    f"La máscara debe tener forma ({self.n_y}, {self.n_x}), "
                    f"recibida {mask.shape}"
                )

        return self._spectrum_extractor.extract_mean_spectrum(mask)

    def get_flux_limits(self) -> Tuple[float, float]:
        """
        Calcula los límites globales de flujo en el cubo.

        Returns:
            Tuple[float, float]: (mínimo, máximo) del flujo. Retorna
                               (NaN, NaN) si no hay datos válidos.

        Example:
            >>> cube = CubeData(np.random.rand(10, 20, 20))
            >>> min_flux, max_flux = cube.get_flux_limits()
            >>> print(f"Rango de flujo: {min_flux:.3f} - {max_flux:.3f}")
        """
        return self._statistics_calculator.calculate_flux_limits()

    def get_statistics_summary(self) -> Dict[str, float]:
        """
        Calcula un resumen estadístico completo del cubo.

        Returns:
            Dict[str, float]: Diccionario con estadísticas principales:
                            min, max, mean, median, std, valid_pixels

        Example:
            >>> cube = CubeData(np.random.rand(10, 20, 20))
            >>> stats = cube.get_statistics_summary()
            >>> print(f"Media: {stats['mean']:.3f}")
            >>> print(f"Desviación estándar: {stats['std']:.3f}")
        """
        return self._statistics_calculator.calculate_statistics_summary()

    def get_coordinate_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre las coordenadas del cubo.

        Returns:
            Dict: Información de coordenadas y dimensiones
        """
        ranges = self._coordinate_manager.get_valid_coordinate_ranges()
        return {
            'shape': self.shape,
            'coordinate_ranges': ranges,
            'wavelength_range': (float(self.wavelength.min()), float(self.wavelength.max())),
            'wavelength_step': float(np.mean(np.diff(self.wavelength))) if len(self.wavelength) > 1 else 0.0
        }

    def as_dict(self) -> Dict[str, Any]:
        """
        Convierte los datos del cubo a un diccionario.

        Returns:
            Dict[str, Any]: Diccionario con todos los datos y metadatos

        Example:
            >>> cube = CubeData(np.random.rand(10, 20, 20))
            >>> data_dict = cube.as_dict()
            >>> print(data_dict.keys())
        """
        return {
            "data": self.data,
            "wavelength": self.wavelength,
            "error": self.error,
            "flag": self.flag,
            "meta": self.meta.copy(),
            "shape": self.shape,
            "coordinate_info": self.get_coordinate_info()
        }

    def validate_integrity(self) -> bool:
        """
        Valida la integridad de todos los datos del cubo.

        Returns:
            bool: True si todos los datos son consistentes

        Raises:
            ValueError: Si se encuentra alguna inconsistencia
        """
        try:
            # Validar estructura básica
            DataValidator.validate_cube_shape(self.data)
            DataValidator.validate_wavelength_consistency(self.data, self.wavelength)

            # Validar datos auxiliares
            if self.error is not None:
                DataValidator.validate_auxiliary_data_shape(self.data, self.error, "error")

            if self.flag is not None:
                DataValidator.validate_auxiliary_data_shape(self.data, self.flag, "flag")

            # Validar que las dimensiones internas sean consistentes
            assert self.n_lambda == self.data.shape[0]
            assert self.n_y == self.data.shape[1]
            assert self.n_x == self.data.shape[2]

            self._logger.info("Validación de integridad completada exitosamente")
            return True

        except Exception as e:
            self._logger.error(f"Error en validación de integridad: {e}")
            raise

    def __repr__(self) -> str:
        """
        Representación string del objeto.

        Returns:
            str: Descripción del cubo de datos
        """
        return (f"CubeData(shape={self.shape}, "
                f"wavelength_range=({self.wavelength.min():.1f}, {self.wavelength.max():.1f}), "
                f"has_error={self.error is not None}, "
                f"has_flag={self.flag is not None})")

    def __str__(self) -> str:
        """
        Representación string legible del objeto.

        Returns:
            str: Descripción detallada del cubo
        """
        stats = self.get_statistics_summary()
        return (f"Cubo Espectral 3D:\n"
                f"  Dimensiones: {self.shape} (lambda, y, x)\n"
                f"  Rango de wavelength: {self.wavelength.min():.1f} - {self.wavelength.max():.1f}\n"
                f"  Rango de flujo: {stats['min']:.2e} - {stats['max']:.2e}\n"
                f"  Píxeles válidos: {stats['valid_pixels']}\n"
                f"  Error incluido: {'Sí' if self.error is not None else 'No'}\n"
                f"  Flags incluidos: {'Sí' if self.flag is not None else 'No'}")