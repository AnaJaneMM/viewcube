import numpy as np
from scipy.spatial import distance
from typing import Tuple, List, Union


def image_quadrant(image_shape: Tuple[int, int], center: Optional[Tuple[int, int]] = None,
                   chunk_factor: float = 3.0) -> Tuple[slice, slice]:
    """Calcula los límites de un cuadrante centrado en una imagen"""
    ny, nx = image_shape
    cy, cx = (ny // 2, nx // 2) if center is None else center

    dy = int(ny / chunk_factor)
    dx = int(nx / chunk_factor)
    return (
        slice(max(0, cy - dy // 2), min(ny, cy + dy // 2)),
        slice(max(0, cx - dx // 2), min(nx, cx + dx // 2))
    )


def image_max_pixel(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[int, int]:
    """Encuentra el píxel con valor máximo considerando máscara"""
    if mask is not None:
        image = np.ma.masked_array(image, mask=~mask)
    return np.unravel_index(np.nanargmax(image), image.shape)


def get_radius(x: np.ndarray, y: np.ndarray) -> float:
    """Calcula radio característico de distribución de puntos"""
    points = np.column_stack((x, y))
    dists = distance.pdist(points)
    return np.min(dists[dists > 0]) if np.any(dists > 0) else 0.0


def rotate_positions(x: np.ndarray, y: np.ndarray, angle: float,
                     ref_point: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Rota coordenadas alrededor de un punto de referencia"""
    theta = np.radians(angle)
    xc, yc = x - ref_point[0], y - ref_point[1]
    return (
        xc * np.cos(theta) - yc * np.sin(theta) + ref_point[0],
        xc * np.sin(theta) + yc * np.cos(theta) + ref_point[1]
    )


def create_hexagon(x: float, y: float, scale: float = 1.0) -> np.ndarray:
    """Genera coordenadas de hexágono regular"""
    sqrt3 = np.sqrt(3)
    return np.array([
        [x - scale, y],
        [x - scale / 2, y + (sqrt3 * scale) / 2],
        [x + scale / 2, y + (sqrt3 * scale) / 2],
        [x + scale, y],
        [x + scale / 2, y - (sqrt3 * scale) / 2],
        [x - scale / 2, y - (sqrt3 * scale) / 2]
    ])


def calculate_spaxel_limits(x: np.ndarray, y: np.ndarray, radius: float,
                            padding_factor: float = 1.2) -> Tuple[float, float, float, float]:
    """Calcula límites de visualización para spaxels"""
    x_center, y_center = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    x_range = (x.max() - x.min()) * padding_factor
    y_range = (y.max() - y.min()) * padding_factor
    return (
        x_center - x_range / 2, x_center + x_range / 2,
        y_center - y_range / 2, y_center + y_range / 2
    )


def generate_rectangle_coords(x: Union[float, np.ndarray], y: Union[float, np.ndarray],
                              width: float) -> Tuple[np.ndarray, np.ndarray]:
    """Genera coordenadas para rectángulos centrados"""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    hw = width / 2
    return (
        np.column_stack([x - hw, x - hw, x + hw, x + hw]),
        np.column_stack([y - hw, y + hw, y + hw, y - hw])
    )