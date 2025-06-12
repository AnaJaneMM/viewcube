"""Utilidades para manejo de mapas de colores y normalización."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from typing import Tuple, Optional, Union, List
from ..rgbmpl import rnorm, IntColorMap, zscale


class ColormapUtils:
    """Utilidades para manejo de mapas de colores y normalización."""

    @staticmethod
    def get_normalization(norm_type: str = 'sqrt', vmin: float = None,
                          vmax: float = None, data: np.ndarray = None,
                          **kwargs) -> Normalize:
        """
        Obtiene objeto de normalización.

        Parameters:
        -----------
        norm_type : str
            Tipo de normalización: 'linear', 'sqrt', 'log', 'ilog', 'asinh', 'power'
        vmin, vmax : float
            Límites de normalización
        data : np.ndarray
            Datos para calcular límites automáticamente
        **kwargs
            Parámetros adicionales para rnorm

        Returns:
        --------
        Normalize
            Objeto de normalización configurado
        """
        # Calcular límites automáticamente si no se proporcionan
        if data is not None and (vmin is None or vmax is None):
            if norm_type == 'zscale' or kwargs.get('zsc', False):
                calc_vmin, calc_vmax = zscale(data)
                vmin = vmin if vmin is not None else calc_vmin
                vmax = vmax if vmax is not None else calc_vmax
            else:
                calc_vmin, calc_vmax = np.nanmin(data), np.nanmax(data)
                vmin = vmin if vmin is not None else calc_vmin
                vmax = vmax if vmax is not None else calc_vmax

        # Crear objeto de normalización
        if norm_type in ['linear', 'sqrt', 'log', 'ilog', 'asinh', 'power']:
            return rnorm(
                scale=norm_type,
                vmin=vmin,
                vmax=vmax,
                zsc=data if kwargs.get('zsc', False) else None,
                **kwargs
            )
        else:
            # Normalización estándar de matplotlib
            return Normalize(vmin=vmin, vmax=vmax)

    @staticmethod
    def create_interactive_colormap(image_plot, data: np.ndarray = None,
                                    enable_dynamic_range: bool = True,
                                    enable_colormap_change: bool = True,
                                    **kwargs) -> IntColorMap:
        """
        Crea mapa de colores interactivo.

        Parameters:
        -----------
        image_plot
            Plot de imagen de matplotlib
        data : np.ndarray
            Datos para el mapa de colores
        enable_dynamic_range : bool
            Habilitar cambio de rango dinámico
        enable_colormap_change : bool
            Habilitar cambio de mapa de colores
        **kwargs
            Parámetros adicionales para IntColorMap

        Returns:
        --------
        IntColorMap
            Objeto de mapa de colores interactivo
        """
        return IntColorMap(
            im=image_plot,
            data=data,
            cdrcm=enable_dynamic_range,
            ccms=enable_colormap_change,
            **kwargs
        )

    @staticmethod
    def apply_scaling(data: np.ndarray, scaling_type: str = 'sqrt',
                      vmin: float = None, vmax: float = None,
                      **kwargs) -> np.ndarray:
        """
        Aplica escalado a los datos.

        Parameters:
        -----------
        data : np.ndarray
            Datos a escalar
        scaling_type : str
            Tipo de escalado
        vmin, vmax : float
            Límites para el escalado
        **kwargs
            Parámetros adicionales

        Returns:
        --------
        np.ndarray
            Datos escalados
        """
        from ..rgbmpl import isfun

        return isfun(
            data,
            scale=scaling_type,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )

    @staticmethod
    def get_color_limits(data: np.ndarray, method: str = 'minmax',
                         percentiles: Tuple[float, float] = (1, 99)) -> Tuple[float, float]:
        """
        Calcula límites de color para visualización.

        Parameters:
        -----------
        data : np.ndarray
            Datos para calcular límites
        method : str
            Método: 'minmax', 'zscale', 'percentile', 'sigma'
        percentiles : Tuple[float, float]
            Percentiles para método 'percentile'

        Returns:
        --------
        Tuple[float, float]
            Límites mínimo y máximo
        """
        # Filtrar valores finitos
        finite_data = data[np.isfinite(data)]

        if len(finite_data) == 0:
            return 0.0, 1.0

        if method == 'minmax':
            return float(np.min(finite_data)), float(np.max(finite_data))

        elif method == 'zscale':
            return zscale(data)

        elif method == 'percentile':
            return tuple(np.percentile(finite_data, percentiles))

        elif method == 'sigma':
            mean = np.mean(finite_data)
            std = np.std(finite_data)
            return mean - 3 * std, mean + 3 * std

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def create_custom_colormap(colors: List[str], name: str = 'custom') -> LinearSegmentedColormap:
        """
        Crea un mapa de colores personalizado.

        Parameters:
        -----------
        colors : List[str]
            Lista de colores para el mapa
        name : str
            Nombre del mapa de colores

        Returns:
        --------
        LinearSegmentedColormap
            Mapa de colores personalizado
        """
        return LinearSegmentedColormap.from_list(name, colors)

    @staticmethod
    def adjust_contrast_brightness(data: np.ndarray, contrast: float = 1.0,
                                   brightness: float = 0.0) -> np.ndarray:
        """
        Ajusta contraste y brillo de los datos.

        Parameters:
        -----------
        data : np.ndarray
            Datos originales
        contrast : float
            Factor de contraste (1.0 = sin cambio)
        brightness : float
            Offset de brillo (0.0 = sin cambio)

        Returns:
        --------
        np.ndarray
            Datos ajustados
        """
        return contrast * data + brightness

    @staticmethod
    def histogram_equalization(data: np.ndarray, nbins: int = 256) -> np.ndarray:
        """
        Aplica ecualización de histograma.

        Parameters:
        -----------
        data : np.ndarray
            Datos originales
        nbins : int
            Número de bins para el histograma

        Returns:
        --------
        np.ndarray
            Datos ecualizados
        """
        # Filtrar valores finitos
        finite_data = data[np.isfinite(data)]

        if len(finite_data) == 0:
            return data

        # Calcular histograma
        hist, bins = np.histogram(finite_data, bins=nbins)

        # Calcular función de distribución acumulativa
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalizar

        # Interpolar valores ecualizados
        equalized = np.interp(data, bins[:-1], cdf)

        return equalized
