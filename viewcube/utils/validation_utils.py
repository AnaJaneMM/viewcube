import os
import fnmatch
import numpy as np
from typing import Union, List, Any, Tuple
from astropy.io import fits

def validate_list(input_data: Any, return_none: bool = True,
                 allow_arrays: bool = False) -> Union[List, None]:
    """Normaliza entrada a lista"""
    if input_data is None and return_none:
        return None
    if isinstance(input_data, (list, tuple)):
        return list(input_data)
    if allow_arrays and isinstance(input_data, np.ndarray):
        return input_data.tolist()
    return [input_data]

def list_files(pattern: str, directory: str = '.', full_path: bool = False) -> List[str]:
    """Lista archivos que coinciden con patrón"""
    return sorted([
        os.path.join(directory, f) if full_path else f
        for f in os.listdir(directory)
        if fnmatch.fnmatch(f, pattern)
    ])

def check_file_exists(filepath: str, exit_on_fail: bool = True) -> bool:
    """Verifica existencia de archivo"""
    exists = os.path.exists(filepath)
    if not exists and exit_on_fail:
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    return exists

def get_min_max(data: np.ndarray, mask_value: float = np.nan) -> Tuple[float, float]:
    """Calcula mínimo y máximo válidos"""
    valid = data[np.isfinite(data)]
    return (valid.min(), valid.max()) if valid.size else (mask_value, mask_value)

def validate_fits_extension(filename: str, ext: Union[int, str]) -> bool:
    """Valida existencia de extensión en FITS"""
    with fits.open(filename) as hdul:
        if isinstance(ext, int):
            return ext < len(hdul)
        return any(ext.lower() in hdu.name.lower() for hdu in hdul)