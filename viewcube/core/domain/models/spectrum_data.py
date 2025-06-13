import numpy as np
from typing import Optional, Tuple, Dict, Any, Union


class SpectrumData:
    """
    Modelo de datos para un espectro individual astronómico.

    Esta clase encapsula los datos fundamentales de un espectro astronómico,
    incluyendo longitudes de onda, flujo, errores y flags de calidad, siguiendo
    el principio de responsabilidad única y manteniendo baja complejidad ciclomática.

    Attributes:
        wavelength (np.ndarray): Array de longitudes de onda en Angstroms.
        flux (np.ndarray): Array de valores de flujo.
        error (Optional[np.ndarray]): Array de errores asociados al flujo.
        flag (Optional[np.ndarray]): Array de flags de calidad de datos.
        meta (Dict[str, Any]): Metadatos adicionales del espectro.

    Examples:
        >>> import numpy as np
        >>> wavelength = np.linspace(4000, 7000, 100)
        >>> flux = np.random.normal(1.0, 0.1, 100)
        >>> spectrum = SpectrumData(wavelength, flux)
        >>> print(f"Rango de longitud de onda: {spectrum.get_wavelength_range()}")
        (4000.0, 7000.0)

    Note:
        Los arrays de entrada son automáticamente convertidos a numpy arrays
        para garantizar consistencia en las operaciones numéricas según las
        mejores prácticas de Python [25][26].
    """

    def __init__(
            self,
            wavelength: Union[list, np.ndarray],
            flux: Union[list, np.ndarray],
            error: Optional[Union[list, np.ndarray]] = None,
            flag: Optional[Union[list, np.ndarray]] = None,
            meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Inicializa un objeto SpectrumData con validación robusta de entrada.

        Este método implementa el patrón de validación temprana para detectar
        errores en tiempo de inicialización y garantizar la integridad de los datos [28].

        Args:
            wavelength: Array de longitudes de onda. Debe ser 1D y tener
                       la misma longitud que flux.
            flux: Array de valores de flujo. Debe ser 1D.
            error: Array opcional de errores asociados al flujo.
                  Si se proporciona, debe tener la misma forma que flux.
            flag: Array opcional de flags de calidad (típicamente enteros).
                 Si se proporciona, debe tener la misma forma que flux.
            meta: Diccionario opcional con metadatos adicionales.

        Raises:
            ValueError: Si los arrays no tienen las dimensiones correctas.
            TypeError: Si los datos no son convertibles a arrays numpy.

        Examples:
            >>> wavelength = [4000, 4001, 4002]
            >>> flux = [1.0, 1.1, 0.9]
            >>> error = [0.1, 0.15, 0.08]
            >>> spectrum = SpectrumData(wavelength, flux, error=error)
        """
        # Aplicación del principio de responsabilidad única: delegamos validación
        self.wavelength, self.flux = self._validate_primary_arrays(wavelength, flux)
        self.error = self._validate_optional_array(error, self.flux.shape, "error")
        self.flag = self._validate_optional_array(flag, self.flux.shape, "flag")
        self.meta = meta or {}

    def _validate_primary_arrays(
            self,
            wavelength: Union[list, np.ndarray],
            flux: Union[list, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Valida y convierte los arrays principales (wavelength y flux).

        Método privado que implementa validación específica para arrays críticos,
        reduciendo la complejidad del constructor principal [20][21].

        Args:
            wavelength: Array de longitudes de onda.
            flux: Array de flujo.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays validados y convertidos a float64.

        Raises:
            TypeError: Si no se pueden convertir a arrays numpy.
            ValueError: Si no tienen la misma forma o están vacíos.
        """
        try:
            wave_array = np.asarray(wavelength, dtype=float)
            flux_array = np.asarray(flux, dtype=float)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Error convirtiendo arrays principales a numpy: {e}")

        if wave_array.size == 0 or flux_array.size == 0:
            raise ValueError("Los arrays wavelength y flux no pueden estar vacíos")

        if wave_array.shape != flux_array.shape:
            raise ValueError(
                f"wavelength y flux deben tener la misma forma. "
                f"Recibido wavelength: {wave_array.shape}, flux: {flux_array.shape}"
            )

        return wave_array, flux_array

    def _validate_optional_array(
            self,
            array_data: Optional[Union[list, np.ndarray]],
            target_shape: Tuple[int, ...],
            name: str
    ) -> Optional[np.ndarray]:
        """
        Valida y convierte un array opcional con verificación de forma.

        Método utilitario que implementa validación consistente para arrays
        opcionales, promoviendo reutilización de código [19][22].

        Args:
            array_data: Datos del array a validar (puede ser None).
            target_shape: Forma objetivo que debe tener el array.
            name: Nombre del array para mensajes de error descriptivos.

        Returns:
            np.ndarray or None: Array validado o None si no se proporcionó.

        Raises:
            TypeError: Si no se puede convertir a array numpy.
            ValueError: Si no tiene la forma correcta.
        """
        if array_data is None:
            return None

        try:
            validated_array = np.asarray(array_data)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Error convirtiendo {name} a numpy array: {e}")

        if validated_array.shape != target_shape:
            raise ValueError(
                f"El array {name} debe tener la forma {target_shape}. "
                f"Recibido: {validated_array.shape}"
            )

        return validated_array

    def apply_flag_mask(self, mask_value: Union[int, float] = 0) -> None:
        """
        Aplica una máscara a los datos de flux y error utilizando los flags.

        Los elementos donde flag > mask_value serán enmascarados (marcados como inválidos).
        Esta operación modifica in-place los arrays convirtiéndolos en masked arrays [27].

        Args:
            mask_value: Valor umbral para crear la máscara. Los elementos con
                       flag > mask_value serán enmascarados. Por defecto 0.

        Raises:
            ValueError: Si no hay flags disponibles para crear la máscara.

        Examples:
            >>> spectrum = SpectrumData([1, 2, 3], [1.0, 2.0, 3.0], flag=[0, 1, 0])
            >>> spectrum.apply_flag_mask(mask_value=0)
            >>> print(spectrum.flux.mask)  # [False, True, False]

        Note:
            Una vez aplicada la máscara, los arrays se convierten en masked arrays
            de numpy y las operaciones posteriores respetan automáticamente la máscara [27].
        """
        if self.flag is None:
            raise ValueError(
                "No hay flags disponibles. Proporcione flags en la inicialización "
                "para usar esta funcionalidad."
            )

        mask = self.flag > mask_value
        self.flux = np.ma.array(self.flux, mask=mask)

        if self.error is not None:
            self.error = np.ma.array(self.error, mask=mask)

    def get_flux_range(self) -> Tuple[float, float]:
        """
        Calcula el rango (mínimo, máximo) de los valores de flujo válidos.

        Ignora automáticamente valores NaN, infinitos y enmascarados, proporcionando
        un análisis robusto de los datos espectrales [27].

        Returns:
            Tuple[float, float]: (valor_mínimo, valor_máximo) del flujo.
            Retorna (nan, nan) si no hay valores válidos.

        Examples:
            >>> spectrum = SpectrumData([1, 2, 3], [1.0, np.nan, 3.0])
            >>> min_flux, max_flux = spectrum.get_flux_range()
            >>> print(f"Rango: {min_flux} - {max_flux}")  # Rango: 1.0 - 3.0

        Note:
            Esta función es segura para usar con masked arrays y valores NaN,
            siguiendo las mejores prácticas de análisis de datos científicos [27].
        """
        valid_data = self._extract_valid_flux_data()
        return self._compute_data_range(valid_data)

    def get_wavelength_range(self) -> Tuple[float, float]:
        """
        Calcula el rango (mínimo, máximo) de las longitudes de onda válidas.

        Proporciona información sobre el rango espectral cubierto por los datos,
        ignorando valores no finitos [27].

        Returns:
            Tuple[float, float]: (longitud_onda_mínima, longitud_onda_máxima).
            Retorna (nan, nan) si no hay valores válidos.

        Examples:
            >>> spectrum = SpectrumData([4000, 5000, 6000], [1.0, 2.0, 3.0])
            >>> min_wave, max_wave = spectrum.get_wavelength_range()
            >>> print(f"Rango espectral: {min_wave} - {max_wave} Å")

        Note:
            Las longitudes de onda se asumen en Angstroms por convención
            astronómica estándar [27].
        """
        valid_wavelengths = self.wavelength[np.isfinite(self.wavelength)]
        return self._compute_data_range(valid_wavelengths)

    def _extract_valid_flux_data(self) -> np.ndarray:
        """
        Extrae datos de flujo válidos considerando máscaras y valores NaN.

        Método utilitario que centraliza la lógica de extracción de datos válidos,
        reduciendo duplicación de código [19][22].

        Returns:
            np.ndarray: Array con solo valores válidos de flujo.
        """
        if np.ma.isMaskedArray(self.flux):
            return self.flux.compressed()
        else:
            return self.flux[np.isfinite(self.flux)]

    def _compute_data_range(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Calcula el rango de un array de datos numéricos.

        Método utilitario reutilizable para cálculo de rangos con manejo
        robusto de casos extremos [20][21].

        Args:
            data: Array de datos numéricos válidos.

        Returns:
            Tuple[float, float]: (mínimo, máximo) o (nan, nan) si está vacío.
        """
        if data.size == 0:
            return (np.nan, np.nan)
        return (float(np.min(data)), float(np.max(data)))

    def as_dict(self) -> Dict[str, Any]:
        """
        Convierte todos los datos del espectro a un diccionario.

        Útil para serialización, debugging, análisis o conversión a otros formatos
        de datos científicos [27].

        Returns:
            Dict[str, Any]: Diccionario conteniendo todas las propiedades:
            - 'wavelength': array de longitudes de onda
            - 'flux': array de flujo
            - 'error': array de errores (puede ser None)
            - 'flag': array de flags (puede ser None)
            - 'meta': diccionario de metadatos

        Examples:
            >>> spectrum = SpectrumData([1, 2], [1.0, 2.0], meta={'object': 'star'})
            >>> data_dict = spectrum.as_dict()
            >>> print(data_dict['meta'])  # {'object': 'star'}

        Note:
            Los arrays numpy se mantienen como tales. Para serialización JSON
            completa, puede ser necesario convertirlos a listas [27].
        """
        return {
            "wavelength": self.wavelength,
            "flux": self.flux,
            "error": self.error,
            "flag": self.flag,
            "meta": self.meta.copy()  # Copia defensiva para evitar modificaciones
        }

    def __repr__(self) -> str:
        """
        Representación string informativa del objeto para debugging.

        Proporciona información esencial sobre el espectro de manera concisa,
        siguiendo las convenciones de Python [25][26].

        Returns:
            str: Representación descriptiva del objeto.
        """
        n_points = len(self.wavelength)
        wave_range = self.get_wavelength_range()
        flux_range = self.get_flux_range()

        return (
            f"SpectrumData(n_points={n_points}, "
            f"wavelength_range=({wave_range[0]:.1f}, {wave_range[1]:.1f}), "
            f"flux_range=({flux_range[0]:.2e}, {flux_range[1]:.2e}))"
        )

    def __len__(self) -> int:
        """
        Retorna el número de puntos espectrales.

        Implementa el protocolo de tamaño de Python para integración
        natural con el ecosistema del lenguaje [25][26].

        Returns:
            int: Número de elementos en el espectro.
        """
        return len(self.wavelength)

    def is_valid(self) -> bool:
        """
        Verifica si el espectro contiene datos válidos.

        Método utilitario para validación rápida del estado del objeto,
        útil en pipelines de procesamiento de datos [28].

        Returns:
            bool: True si hay al menos un punto espectral con datos válidos.
        """
        return len(self.wavelength) > 0 and not np.all(np.isnan(self.flux))

    def copy(self) -> 'SpectrumData':
        """
        Crea una copia profunda del objeto SpectrumData.

        Útil para operaciones que modifican los datos sin afectar el original,
        siguiendo patrones inmutables [21][22].

        Returns:
            SpectrumData: Nueva instancia con datos copiados.
        """
        error_copy = self.error.copy() if self.error is not None else None
        flag_copy = self.flag.copy() if self.flag is not None else None

        return SpectrumData(
            wavelength=self.wavelength.copy(),
            flux=self.flux.copy(),
            error=error_copy,
            flag=flag_copy,
            meta=self.meta.copy()
        )