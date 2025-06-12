"""Utilidades para cálculos geométricos."""

import numpy as np
import math
from typing import Tuple, List


class GeometryUtils:
    """Utilidades para cálculos geométricos y transformaciones."""

    @staticmethod
    def calculate_spaxel_limits(x: np.ndarray, y: np.ndarray,
                                radius: float) -> List[float]:
        """Calcula límites para visualización de spaxels."""
        spax_fac = radius * 7
        xbar = abs(max(x) - min(x)) * 0.5
        ybar = abs(max(y) - min(y)) * 0.5
        xmed = xbar + min(x)
        ymed = ybar + min(y)

        xfbar = 1.2 if xbar > spax_fac else 4.0
        yfbar = 1.2 if ybar > spax_fac else 4.0

        # Casos especiales PPAK
        if len(x) == 331 or len(x) == 993:
            yfbar = 1.3
        if len(x) == 382:
            xfbar = 1.3

        xmax_mosaic = round(xmed + xbar * xfbar)
        xmin_mosaic = round(xmed - xbar * xfbar)
        ymax_mosaic = round(ymed + ybar * yfbar)
        ymin_mosaic = round(ymed - ybar * yfbar)

        return [xmin_mosaic, xmax_mosaic, ymin_mosaic, ymax_mosaic]

    @staticmethod
    def calculate_wavelength_limits(wavelength_arrays: List[np.ndarray],
                                    padding_fraction: float = 0.05,
                                    user_limits: Tuple[float, float] = None) -> Tuple[float, float]:
        """Calcula límites de longitud de onda con padding opcional."""
        # Filtrar arrays válidos
        valid_arrays = [wl for wl in wavelength_arrays if wl is not None]

        if not valid_arrays:
            return 0.0, 1.0

        # Encontrar límites globales
        all_mins = [np.min(wl) for wl in valid_arrays]
        all_maxs = [np.max(wl) for wl in valid_arrays]

        wl_min = min(all_mins)
        wl_max = max(all_maxs)

        # Aplicar límites del usuario si se proporcionan
        if user_limits:
            user_min, user_max = user_limits
            if user_min is not None:
                wl_min = user_min
            if user_max is not None:
                wl_max = user_max

        # Añadir padding
        wavelength_range = abs(wl_max - wl_min)
        padding = wavelength_range * padding_fraction

        return wl_min - padding, wl_max + padding

    @staticmethod
    def calculate_flux_limits(flux_limits: Tuple[float, float] = None) -> Tuple[Optional[float], Optional[float]]:
        """Valida y retorna límites de flujo."""
        if flux_limits is None:
            return None, None

        if not isinstance(flux_limits, (list, tuple)) or len(flux_limits) != 2:
            print("Flux limits should be a tuple or list of two items: ex. --> (None, 1e-18)")
            return None, None

        return flux_limits[0], flux_limits[1]

    @staticmethod
    def create_rectangle_vertices(x: float, y: float, size: float) -> Tuple[np.ndarray, np.ndarray]:
        """Crea vértices de rectángulo para visualización."""
        if isinstance(x, (list, tuple)):
            x = np.array(x)
            y = np.array(y)

        if isinstance(x, (int, float)):
            xv = x + np.array([0.0, 0.0, size, size])
            yv = y + np.array([0.0, size, size, 0.0])
        else:
            xv = x[:, np.newaxis] + np.array([0.0, 0.0, size, size])
            yv = y[:, np.newaxis] + np.array([0.0, size, size, 0.0])

        return xv, yv

    @staticmethod
    def pixel_to_world_coordinates(pixel_x: int, pixel_y: int,
                                   reference_x: float, reference_y: float,
                                   spatial_resolution: float) -> Tuple[float, float]:
        """Convierte coordenadas de pixel a coordenadas del mundo."""
        world_x = (pixel_x - reference_x) * spatial_resolution
        world_y = (pixel_y - reference_y) * spatial_resolution
        return world_x, world_y

    @staticmethod
    def world_to_pixel_coordinates(world_x: float, world_y: float,
                                   reference_x: float, reference_y: float,
                                   spatial_resolution: float) -> Tuple[int, int]:
        """Convierte coordenadas del mundo a coordenadas de pixel."""
        pixel_x = int((world_x / spatial_resolution) + reference_x)
        pixel_y = int((world_y / spatial_resolution) + reference_y)
        return pixel_x, pixel_y

    @staticmethod
    def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calcula distancia euclidiana entre dos puntos."""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def find_nearest_spaxel(target_x: float, target_y: float,
                            spaxel_positions: List[Tuple[float, float]],
                            max_distance: float = None) -> Tuple[int, float]:
        """Encuentra el spaxel más cercano a una posición objetivo."""
        if not spaxel_positions:
            return -1, float('inf')

        min_distance = float('inf')
        nearest_index = -1

        for i, (sx, sy) in enumerate(spaxel_positions):
            distance = GeometryUtils.calculate_distance(target_x, target_y, sx, sy)

            if distance < min_distance:
                if max_distance is None or distance <= max_distance:
                    min_distance = distance
                    nearest_index = i

        return nearest_index, min_distance
