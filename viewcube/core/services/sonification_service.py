import os
import sys
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List, Protocol
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import logging

from ..domain.models.spectrum_data import SpectrumData
from ..domain.models.cube_data import CubeData
from ..domain.entities.astronomical_entities import AstronomicalObject

logger = logging.getLogger(__name__)


class SonificationMode(Enum):
    """Modos de sonificación soportados."""
    SPECTRUM = "spectrum"
    CUBE = "cube"
    INTEGRATED = "integrated"
    CONTINUOUS = "continuous"


class AudioFormat(Enum):
    """Formatos de audio soportados."""
    PCM_16 = "pcm_16"
    PCM_24 = "pcm_24"
    PCM_32 = "pcm_32"
    FLOAT_32 = "float_32"


@dataclass
class AudioConfig:
    """Configuración de audio."""
    sample_rate: int = 48000
    buffer_size: int = 1024
    channels: int = 2
    bit_depth: int = 16
    format: AudioFormat = AudioFormat.PCM_16


@dataclass
class SonificationConfig:
    """Configuración de sonificación."""
    frequency_range: Tuple[float, float] = (200, 2000)
    volume_range: Tuple[float, float] = (0.0, 1.0)
    spatial_range: float = 360.0
    latent_dimensions: int = 6
    a_weighting: bool = True
    flux_sensitivity: bool = True
    distance_attenuation: bool = True


@dataclass
class SonificationStatus:
    """Estado del sistema de sonificación."""
    active: bool = False
    sonicube_initialized: bool = False
    flux_sensitive: bool = True
    directory: Optional[str] = None
    port: int = 9970
    current_cube_loaded: bool = False
    continuous_active: bool = False
    autoencoder_loaded: bool = False


class AudioProcessorProtocol(Protocol):
    """Protocolo para procesadores de audio."""

    def process_spectrum(self, spectrum: SpectrumData) -> Dict[str, Any]:
        """Procesa un espectro para sonificación."""
        ...


class SonificationEngineProtocol(Protocol):
    """Protocolo para motores de sonificación."""

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Inicializa el motor."""
        ...

    def sonify(self, data: Any, position: Optional[Tuple[int, int]] = None) -> bool:
        """Sonifica datos."""
        ...


class AudioParameterMapper:
    """Mapeador especializado de parámetros de audio."""

    def __init__(self, config: SonificationConfig):
        """
        Inicializa el mapeador de parámetros.

        Args:
            config: Configuración de sonificación
        """
        self._config = config

    def map_spectrum_to_audio(self, spectrum: SpectrumData) -> Dict[str, Any]:
        """
        Mapea un espectro a parámetros de audio.

        Args:
            spectrum: Datos del espectro

        Returns:
            Diccionario con parámetros de audio
        """
        try:
            # Calcular estadísticas del espectro
            stats = self._calculate_spectrum_statistics(spectrum)

            # Mapear frecuencias
            frequencies = self._map_wavelength_to_frequency(spectrum.wavelength)

            # Mapear amplitudes
            amplitudes = self._map_flux_to_amplitude(spectrum.flux, stats)

            # Aplicar sensibilidad de flujo si está habilitada
            if self._config.flux_sensitivity:
                amplitudes = self._apply_flux_sensitivity(amplitudes, stats)

            return {
                'frequencies': frequencies,
                'amplitudes': amplitudes,
                'duration': 1.0,
                'statistics': stats,
                'mapping_config': self._config
            }

        except Exception as e:
            logger.error(f"Error mapeando espectro a audio: {e}")
            return {}

    def _calculate_spectrum_statistics(self, spectrum: SpectrumData) -> Dict[str, float]:
        """
        Calcula estadísticas del espectro para mapeo.

        Args:
            spectrum: Datos del espectro

        Returns:
            Diccionario con estadísticas
        """
        valid_flux = spectrum.flux[np.isfinite(spectrum.flux)]

        if len(valid_flux) == 0:
            return {
                'mean': 0.0, 'median': 0.0, 'std': 0.0,
                'min': 0.0, 'max': 0.0, 'range': 0.0
            }

        return {
            'mean': float(np.mean(valid_flux)),
            'median': float(np.median(valid_flux)),
            'std': float(np.std(valid_flux)),
            'min': float(np.min(valid_flux)),
            'max': float(np.max(valid_flux)),
            'range': float(np.max(valid_flux) - np.min(valid_flux))
        }

    def _map_wavelength_to_frequency(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Mapea longitudes de onda a frecuencias de audio.

        Args:
            wavelength: Array de longitudes de onda

        Returns:
            Array de frecuencias de audio
        """
        if len(wavelength) == 0:
            return np.array([])

        freq_min, freq_max = self._config.frequency_range
        wave_min, wave_max = np.min(wavelength), np.max(wavelength)

        if wave_max == wave_min:
            return np.full_like(wavelength, freq_min)

        # Mapeo inverso: wavelengths más cortas -> frecuencias más altas
        normalized = (wave_max - wavelength) / (wave_max - wave_min)
        frequencies = freq_min + (freq_max - freq_min) * normalized

        return frequencies

    def _map_flux_to_amplitude(self, flux: np.ndarray,
                               stats: Dict[str, float]) -> np.ndarray:
        """
        Mapea flujo a amplitudes de audio.

        Args:
            flux: Array de flujo
            stats: Estadísticas del espectro

        Returns:
            Array de amplitudes
        """
        if len(flux) == 0:
            return np.array([])

        vol_min, vol_max = self._config.volume_range

        if stats['range'] > 0:
            normalized = (flux - stats['min']) / stats['range']
            amplitudes = vol_min + (vol_max - vol_min) * normalized
        else:
            amplitudes = np.full_like(flux, vol_min)

        # Asegurar que las amplitudes estén en rango válido
        return np.clip(amplitudes, vol_min, vol_max)

    def _apply_flux_sensitivity(self, amplitudes: np.ndarray,
                                stats: Dict[str, float]) -> np.ndarray:
        """
        Aplica factor de sensibilidad al flujo.

        Args:
            amplitudes: Amplitudes originales
            stats: Estadísticas del espectro

        Returns:
            Amplitudes modificadas
        """
        if stats['max'] == 0:
            return amplitudes

        flux_factor = np.clip(stats['median'] / (stats['max'] + 1e-10), 0.1, 1.0)
        return amplitudes * flux_factor


class SoniCubeInterface:
    """Interfaz simplificada para SoniCube."""

    def __init__(self, sonicube_directory: str, port: int = 9970):
        """
        Inicializa la interfaz de SoniCube.

        Args:
            sonicube_directory: Directorio de SoniCube
            port: Puerto para comunicación OSC
        """
        self.sonicube_directory = sonicube_directory
        self.port = port
        self._sonicube_class = None
        self._instance = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Inicializa SoniCube.

        Returns:
            True si la inicialización fue exitosa
        """
        try:
            # Añadir directorio al path
            if self.sonicube_directory not in sys.path:
                sys.path.append(self.sonicube_directory)

            # Importar SoniCube
            from sonicube import SoniCube
            self._sonicube_class = SoniCube
            self._initialized = True

            logger.info("SoniCube importado correctamente")
            return True

        except ImportError as e:
            logger.error(f"No se pudo importar SoniCube: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inicializando SoniCube: {e}")
            return False

    def create_instance(self, fits_file: str, cube_data: CubeData,
                        matplotlib_figure=None,
                        reference_position: Optional[Tuple[int, int]] = None,
                        flux_sensitive: bool = True) -> bool:
        """
        Crea una instancia de SoniCube.

        Args:
            fits_file: Archivo FITS original
            cube_data: Datos del cubo
            matplotlib_figure: Figura de matplotlib
            reference_position: Posición de referencia
            flux_sensitive: Sensibilidad al flujo

        Returns:
            True si la instancia se creó exitosamente
        """
        if not self._initialized:
            logger.error("SoniCube no está inicializado")
            return False

        try:
            self._instance = self._sonicube_class(
                fig=matplotlib_figure,
                file=fits_file,
                base_dir=self.sonicube_directory,
                port=self.port,
                ref=reference_position,
                data=cube_data.data,
                flux_sensitive=flux_sensitive
            )

            logger.info("Instancia de SoniCube creada exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error creando instancia de SoniCube: {e}")
            return False

    def sonify_spaxel(self, x: int, y: int, flux: np.ndarray) -> bool:
        """
        Sonifica un spaxel específico.

        Args:
            x: Coordenada X
            y: Coordenada Y
            flux: Array de flujo

        Returns:
            True si la sonificación fue exitosa
        """
        if not self._instance:
            logger.warning("Instancia de SoniCube no disponible")
            return False

        try:
            self._instance.sonify(y, x, flux)
            return True
        except Exception as e:
            logger.error(f"Error en sonificación de spaxel: {e}")
            return False

    def stop_sound(self) -> None:
        """Detiene la sonificación actual."""
        if self._instance:
            try:
                self._instance.stop_sound()
            except Exception as e:
                logger.warning(f"Error deteniendo sonificación: {e}")

    def close(self) -> None:
        """Cierra y limpia la instancia."""
        if self._instance:
            try:
                if hasattr(self._instance, 'close_sound'):
                    self._instance.close_sound()
            except Exception as e:
                logger.warning(f"Error cerrando SoniCube: {e}")
            finally:
                self._instance = None

    def is_available(self) -> bool:
        """Verifica si SoniCube está disponible."""
        return self._instance is not None

    def get_flux_sensitivity(self) -> bool:
        """Obtiene la configuración de sensibilidad al flujo."""
        if self._instance:
            return getattr(self._instance, 'flux_sensitive', True)
        return True

    def set_flux_sensitivity(self, enabled: bool) -> None:
        """Establece la sensibilidad al flujo."""
        if self._instance:
            self._instance.flux_sensitive = enabled


class SyntheticAudioGenerator:
    """Generador de audio sintético para pruebas."""

    def __init__(self, config: AudioConfig):
        """
        Inicializa el generador de audio sintético.

        Args:
            config: Configuración de audio
        """
        self._config = config

    def create_synthetic_sonification(self, wavelength: np.ndarray,
                                      flux_pattern: str = 'sine',
                                      frequency: float = 440.0,
                                      amplitude: float = 1.0,
                                      duration: float = 2.0) -> Dict[str, Any]:
        """
        Crea una sonificación sintética para pruebas.

        Args:
            wavelength: Array de longitudes de onda
            flux_pattern: Tipo de patrón ('sine', 'noise', 'chirp')
            frequency: Frecuencia base en Hz
            amplitude: Amplitud del patrón
            duration: Duración en segundos

        Returns:
            Diccionario con datos de sonificación sintética
        """
        n_points = len(wavelength)

        # Generar patrón de flujo
        if flux_pattern == 'sine':
            t = np.linspace(0, duration, n_points)
            flux = amplitude * np.sin(2 * np.pi * frequency * t)
        elif flux_pattern == 'noise':
            flux = amplitude * np.random.normal(0, 1, n_points)
        elif flux_pattern == 'chirp':
            t = np.linspace(0, duration, n_points)
            flux = amplitude * np.sin(2 * np.pi * frequency * t * t / duration)
        else:
            flux = np.ones(n_points) * amplitude

        # Crear espectro sintético
        synthetic_spectrum = SpectrumData(
            wavelength=wavelength,
            flux=flux,
            meta={'synthetic': True, 'pattern': flux_pattern}
        )

        return {
            'spectrum': synthetic_spectrum,
            'pattern': flux_pattern,
            'duration': duration,
            'audio_config': self._config
        }


class ContinuousSonificationManager:
    """Gestor para sonificación continua."""

    def __init__(self, sonification_interface: SoniCubeInterface):
        """
        Inicializa el gestor de sonificación continua.

        Args:
            sonification_interface: Interfaz de sonificación
        """
        self._interface = sonification_interface
        self._active = False
        self._thread = None
        self._stop_event = threading.Event()

    def start_continuous(self, spectrum_generator,
                         position_generator=None,
                         interval: float = 0.1) -> bool:
        """
        Inicia sonificación continua.

        Args:
            spectrum_generator: Generador de espectros
            position_generator: Generador de posiciones
            interval: Intervalo entre sonificaciones

        Returns:
            True si se inició exitosamente
        """
        if self._active:
            logger.warning("Sonificación continua ya está activa")
            return False

        if not self._interface.is_available():
            logger.error("Interfaz de sonificación no disponible")
            return False

        self._active = True
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._continuous_loop,
            args=(spectrum_generator, position_generator, interval),
            daemon=True
        )
        self._thread.start()

        logger.info("Sonificación continua iniciada")
        return True

    def stop_continuous(self) -> None:
        """Detiene la sonificación continua."""
        if not self._active:
            return

        self._active = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._interface.stop_sound()
        logger.info("Sonificación continua detenida")

    def _continuous_loop(self, spectrum_generator, position_generator, interval: float) -> None:
        """Loop interno para sonificación continua."""
        while self._active and not self._stop_event.is_set():
            try:
                # Obtener siguiente espectro
                spectrum = next(spectrum_generator, None)
                if spectrum is None:
                    break

                # Obtener posición si hay generador
                position = (0, 0)  # Posición por defecto
                if position_generator:
                    pos = next(position_generator, None)
                    if pos:
                        position = pos

                # Sonificar
                self._interface.sonify_spaxel(position[0], position[1], spectrum.flux)

                # Esperar intervalo
                if self._stop_event.wait(interval):
                    break

            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error en sonificación continua: {e}")
                break

        self._active = False

    def is_active(self) -> bool:
        """Verifica si la sonificación continua está activa."""
        return self._active


class SonificationService:
    """
    Servicio de sonificación completamente refactorizado.

    Esta implementación separa responsabilidades, reduce la complejidad
    y mejora la modularidad manteniendo toda la funcionalidad original.
    """

    def __init__(self,
                 sonification_directory: Optional[str] = None,
                 auto_start: bool = False,
                 default_port: int = 9970):
        """
        Inicializa el servicio de sonificación con arquitectura modular.

        Args:
            sonification_directory: Directorio base para sonificación
            auto_start: Iniciar automáticamente el motor de sonificación
            default_port: Puerto por defecto para comunicación OSC
        """
        # Configuración básica
        self.sonification_directory = sonification_directory
        self.default_port = default_port
        self.auto_start = auto_start

        # Configuraciones
        self._audio_config = AudioConfig()
        self._sonification_config = SonificationConfig()

        # Componentes especializados
        self._parameter_mapper = AudioParameterMapper(self._sonification_config)
        self._sonicube_interface = None
        self._synthetic_generator = SyntheticAudioGenerator(self._audio_config)
        self._continuous_manager = None

        # Estado del servicio
        self._status = SonificationStatus(port=default_port, directory=sonification_directory)
        self._current_cube_data = None

        # Inicializar si está configurado
        if auto_start and sonification_directory:
            self.initialize_sonification()

        logger.info("SonificationService inicializado con arquitectura modular")

    # === MÉTODOS PÚBLICOS DE LA INTERFAZ ORIGINAL ===

    def initialize_sonification(self) -> bool:
        """
        Inicializa el sistema de sonificación.

        Returns:
            True si la inicialización fue exitosa
        """
        if not self.sonification_directory:
            logger.warning("Directorio de sonificación no especificado")
            return False

        if not os.path.exists(self.sonification_directory):
            logger.error(f"Directorio no existe: {self.sonification_directory}")
            return False

        try:
            # Inicializar interfaz de SoniCube
            self._sonicube_interface = SoniCubeInterface(
                self.sonification_directory, self.default_port
            )

            success = self._sonicube_interface.initialize()
            if success:
                self._continuous_manager = ContinuousSonificationManager(
                    self._sonicube_interface
                )
                self._status.active = True
                self._status.sonicube_initialized = True

                logger.info("Sistema de sonificación inicializado correctamente")
                return True
            else:
                logger.error("Falló la inicialización de SoniCube")
                return False

        except Exception as e:
            logger.error(f"Error inicializando sonificación: {e}")
            return False

    def is_active(self) -> bool:
        """
        Verifica si el sistema de sonificación está activo.

        Returns:
            True si está activo
        """
        return self._status.active

    def setup_cube_sonification(self,
                                cube_data: CubeData,
                                fits_file: str,
                                reference_position: Optional[Tuple[int, int]] = None,
                                matplotlib_figure=None) -> bool:
        """
        Configura la sonificación para un cubo de datos.

        Args:
            cube_data: Datos del cubo
            fits_file: Archivo FITS original
            reference_position: Posición de referencia (y, x)
            matplotlib_figure: Figura de matplotlib para eventos

        Returns:
            True si la configuración fue exitosa
        """
        if not self._status.active or not self._sonicube_interface:
            logger.warning("Sistema de sonificación no está activo")
            return False

        try:
            # Almacenar datos del cubo
            self._current_cube_data = cube_data

            # Crear instancia de SoniCube
            success = self._sonicube_interface.create_instance(
                fits_file=fits_file,
                cube_data=cube_data,
                matplotlib_figure=matplotlib_figure,
                reference_position=reference_position,
                flux_sensitive=self._status.flux_sensitive
            )

            if success:
                self._status.current_cube_loaded = True
                logger.info("Sonificación de cubo configurada correctamente")
                return True
            else:
                logger.error("Error configurando sonificación de cubo")
                return False

        except Exception as e:
            logger.error(f"Error configurando sonificación de cubo: {e}")
            return False

    def sonify_spaxel(self, x: int, y: int, spectrum: Optional[SpectrumData] = None) -> bool:
        """
        Sonifica un spaxel específico.

        Args:
            x: Coordenada X del spaxel
            y: Coordenada Y del spaxel
            spectrum: Datos del espectro (opcional)

        Returns:
            True si la sonificación fue exitosa
        """
        if not self._status.active or not self._sonicube_interface:
            logger.warning("Sistema de sonificación no disponible")
            return False

        try:
            # Obtener espectro si no se proporciona
            if spectrum is None and self._current_cube_data:
                spectrum = self._current_cube_data.get_spaxel_spectrum(x, y)

            if spectrum is None:
                logger.warning(f"No se pudo obtener espectro para spaxel ({x}, {y})")
                return False

            # Sonificar usando SoniCube
            return self._sonicube_interface.sonify_spaxel(x, y, spectrum.flux)

        except Exception as e:
            logger.error(f"Error sonificando spaxel ({x}, {y}): {e}")
            return False

    def sonify_spectrum(self, spectrum: SpectrumData,
                        position: Optional[Tuple[int, int]] = None) -> bool:
        """
        Sonifica un espectro independiente.

        Args:
            spectrum: Datos del espectro
            position: Posición opcional para espacialización

        Returns:
            True si la sonificación fue exitosa
        """
        if not self._status.active or not self._sonicube_interface:
            logger.warning("Sistema de sonificación no disponible")
            return False

        try:
            # Usar posición por defecto si no se proporciona
            pos = position or (0, 0)

            # Sonificar usando SoniCube
            return self._sonicube_interface.sonify_spaxel(pos[0], pos[1], spectrum.flux)

        except Exception as e:
            logger.error(f"Error sonificando espectro: {e}")
            return False

    def sonify_integrated_spectrum(self, cube_data: CubeData,
                                   mask: Optional[np.ndarray] = None) -> bool:
        """
        Sonifica el espectro integrado de un cubo.

        Args:
            cube_data: Datos del cubo
            mask: Máscara opcional para seleccionar spaxels

        Returns:
            True si la sonificación fue exitosa
        """
        try:
            # Calcular espectro integrado
            integrated_spectrum = cube_data.get_mean_spectrum(mask)

            # Sonificar espectro integrado
            return self.sonify_spectrum(integrated_spectrum, position=(0, 0))

        except Exception as e:
            logger.error(f"Error sonificando espectro integrado: {e}")
            return False

    def start_continuous_sonification(self, cube_data: CubeData,
                                      path_generator,
                                      interval: float = 0.1) -> bool:
        """
        Inicia sonificación continua a lo largo de un camino.

        Args:
            cube_data: Datos del cubo
            path_generator: Generador que produce coordenadas (x, y)
            interval: Intervalo entre sonificaciones en segundos

        Returns:
            True si se inició exitosamente
        """
        if not self._continuous_manager:
            logger.error("Gestor de sonificación continua no disponible")
            return False

        # Crear generador de espectros
        def spectrum_generator():
            for x, y in path_generator:
                yield cube_data.get_spaxel_spectrum(x, y)

        # Crear generador de posiciones
        def position_generator():
            for x, y in path_generator:
                yield (x, y)

        return self._continuous_manager.start_continuous(
            spectrum_generator(), position_generator(), interval
        )

    def stop_continuous_sonification(self) -> None:
        """Detiene la sonificación continua."""
        if self._continuous_manager:
            self._continuous_manager.stop_continuous()

    def is_continuous_active(self) -> bool:
        """
        Verifica si la sonificación continua está activa.

        Returns:
            True si está activa
        """
        if self._continuous_manager:
            return self._continuous_manager.is_active()
        return False

    def stop_sound(self) -> None:
        """Detiene toda sonificación activa."""
        if self._sonicube_interface:
            self._sonicube_interface.stop_sound()

        if self._continuous_manager and self._continuous_manager.is_active():
            self._continuous_manager.stop_continuous()

    def toggle_flux_sensitivity(self) -> bool:
        """
        Alterna la sensibilidad al flujo.

        Returns:
            Nuevo estado de sensibilidad al flujo
        """
        current_state = self.get_flux_sensitivity()
        new_state = not current_state
        self.set_flux_sensitivity(new_state)
        return new_state

    def get_flux_sensitivity(self) -> bool:
        """
        Obtiene el estado actual de sensibilidad al flujo.

        Returns:
            True si está habilitada
        """
        if self._sonicube_interface:
            return self._sonicube_interface.get_flux_sensitivity()
        return self._status.flux_sensitive

    def set_flux_sensitivity(self, enabled: bool) -> None:
        """
        Establece la sensibilidad al flujo.

        Args:
            enabled: True para habilitar sensibilidad al flujo
        """
        self._status.flux_sensitive = enabled
        if self._sonicube_interface:
            self._sonicube_interface.set_flux_sensitivity(enabled)

        logger.info(f"Sensibilidad al flujo {'habilitada' if enabled else 'deshabilitada'}")

    def get_sonification_status(self) -> SonificationStatus:
        """
        Obtiene el estado completo del sistema de sonificación.

        Returns:
            Objeto SonificationStatus con el estado actual
        """
        # Actualizar estado dinámico
        if self._continuous_manager:
            self._status.continuous_active = self._continuous_manager.is_active()

        return self._status

    def create_synthetic_sonification(self, wavelength: np.ndarray,
                                      pattern: str = 'sine',
                                      frequency: float = 440.0) -> Dict[str, Any]:
        """
        Crea una sonificación sintética para pruebas.

        Args:
            wavelength: Array de longitudes de onda
            pattern: Tipo de patrón ('sine', 'noise', 'chirp')
            frequency: Frecuencia base en Hz

        Returns:
            Diccionario con datos de sonificación sintética
        """
        return self._synthetic_generator.create_synthetic_sonification(
            wavelength, pattern, frequency
        )

    def test_sonification(self, pattern: str = 'sine') -> bool:
        """
        Ejecuta una prueba básica de sonificación.

        Args:
            pattern: Patrón de prueba

        Returns:
            True si la prueba fue exitosa
        """
        try:
            # Crear datos sintéticos de prueba
            wavelength = np.linspace(4000, 8000, 1000)
            synthetic_data = self.create_synthetic_sonification(wavelength, pattern)

            # Sonificar espectro de prueba
            return self.sonify_spectrum(synthetic_data['spectrum'])

        except Exception as e:
            logger.error(f"Error en prueba de sonificación: {e}")
            return False

    def configure_audio(self, **kwargs) -> None:
        """
        Configura parámetros de audio.

        Args:
            **kwargs: Parámetros de configuración de audio
        """
        for key, value in kwargs.items():
            if hasattr(self._audio_config, key):
                setattr(self._audio_config, key, value)
                logger.debug(f"Configuración de audio actualizada: {key}={value}")

    def configure_sonification(self, **kwargs) -> None:
        """
        Configura parámetros de sonificación.

        Args:
            **kwargs: Parámetros de configuración de sonificación
        """
        for key, value in kwargs.items():
            if hasattr(self._sonification_config, key):
                setattr(self._sonification_config, key, value)
                logger.debug(f"Configuración de sonificación actualizada: {key}={value}")

    def get_audio_config(self) -> AudioConfig:
        """
        Obtiene la configuración actual de audio.

        Returns:
            Objeto AudioConfig con la configuración actual
        """
        return self._audio_config

    def get_sonification_config(self) -> SonificationConfig:
        """
        Obtiene la configuración actual de sonificación.

        Returns:
            Objeto SonificationConfig con la configuración actual
        """
        return self._sonification_config

    def get_available_patterns(self) -> List[str]:
        """
        Obtiene los patrones de sonificación disponibles.

        Returns:
            Lista de patrones disponibles
        """
        return ['sine', 'noise', 'chirp', 'constant']

    def calculate_audio_parameters(self, spectrum: SpectrumData) -> Dict[str, Any]:
        """
        Calcula parámetros de audio para un espectro sin sonificar.

        Args:
            spectrum: Datos del espectro

        Returns:
            Diccionario con parámetros de audio calculados
        """
        return self._parameter_mapper.map_spectrum_to_audio(spectrum)

    def cleanup(self) -> None:
        """Limpia recursos y cierra conexiones."""
        try:
            # Detener sonificación continua
            if self._continuous_manager and self._continuous_manager.is_active():
                self._continuous_manager.stop_continuous()

            # Detener sonido actual
            if self._sonicube_interface:
                self._sonicube_interface.stop_sound()
                self._sonicube_interface.close()

            # Resetear estado
            self._status.active = False
            self._status.sonicube_initialized = False
            self._status.current_cube_loaded = False
            self._status.continuous_active = False

            self._current_cube_data = None
            self._sonicube_interface = None
            self._continuous_manager = None

            logger.info("SonificationService limpiado correctamente")

        except Exception as e:
            logger.error(f"Error durante limpieza: {e}")

    def __del__(self):
        """Destructor que asegura la limpieza de recursos."""
        try:
            self.cleanup()
        except Exception:
            pass  # Evitar errores en destructor


# Función de utilidad para crear el servicio
def create_sonification_service(sonification_directory: Optional[str] = None,
                                auto_start: bool = False,
                                default_port: int = 9970) -> SonificationService:
    """
    Crea y configura una instancia de SonificationService.

    Args:
        sonification_directory: Directorio base para sonificación
        auto_start: Iniciar automáticamente el motor
        default_port: Puerto por defecto para OSC

    Returns:
        SonificationService configurado
    """
    service = SonificationService(
        sonification_directory=sonification_directory,
        auto_start=auto_start,
        default_port=default_port
    )

    logger.info("SonificationService creado exitosamente")
    return service


# Clase de pruebas integradas
class SonificationServiceTester:
    """Clase de pruebas para el SonificationService."""

    def __init__(self, service: SonificationService):
        """
        Inicializa el tester.

        Args:
            service: Servicio de sonificación a probar
        """
        self.service = service
        self.test_results = []

    def run_all_tests(self) -> bool:
        """
        Ejecuta todas las pruebas del servicio.

        Returns:
            True si todas las pruebas pasaron
        """
        tests = [
            self._test_service_initialization,
            self._test_configuration_management,
            self._test_synthetic_audio_generation,
            self._test_parameter_mapping,
            self._test_status_reporting,
            self._test_flux_sensitivity_toggle,
            self._test_cleanup
        ]

        all_passed = True
        for test in tests:
            try:
                passed, message = test()
                self.test_results.append((test.__name__, passed, message))
                if not passed:
                    all_passed = False
                    print(f"❌ {test.__name__}: {message}")
                else:
                    print(f"✅ {test.__name__}: {message}")
            except Exception as e:
                all_passed = False
                error_msg = f"Test error: {str(e)}"
                self.test_results.append((test.__name__, False, error_msg))
                print(f"💥 {test.__name__}: {error_msg}")

        return all_passed

    def _test_service_initialization(self) -> Tuple[bool, str]:
        """Prueba la inicialización del servicio."""
        if not isinstance(self.service, SonificationService):
            return False, "Servicio no es instancia de SonificationService"

        if not hasattr(self.service, '_parameter_mapper'):
            return False, "Mapeador de parámetros no inicializado"

        if not hasattr(self.service, '_synthetic_generator'):
            return False, "Generador sintético no inicializado"

        return True, "Servicio inicializado correctamente"

    def _test_configuration_management(self) -> Tuple[bool, str]:
        """Prueba la gestión de configuraciones."""
        try:
            # Probar configuración de audio
            original_sample_rate = self.service.get_audio_config().sample_rate
            self.service.configure_audio(sample_rate=44100)
            new_sample_rate = self.service.get_audio_config().sample_rate

            if new_sample_rate != 44100:
                return False, "Configuración de audio no se aplicó"

            # Probar configuración de sonificación
            original_freq_range = self.service.get_sonification_config().frequency_range
            self.service.configure_sonification(frequency_range=(100, 1000))
            new_freq_range = self.service.get_sonification_config().frequency_range

            if new_freq_range != (100, 1000):
                return False, "Configuración de sonificación no se aplicó"

            return True, "Gestión de configuraciones exitosa"

        except Exception as e:
            return False, f"Error en configuraciones: {e}"

    def _test_synthetic_audio_generation(self) -> Tuple[bool, str]:
        """Prueba la generación de audio sintético."""
        try:
            wavelength = np.linspace(4000, 8000, 100)

            for pattern in self.service.get_available_patterns():
                synthetic_data = self.service.create_synthetic_sonification(
                    wavelength, pattern
                )

                if 'spectrum' not in synthetic_data:
                    return False, f"Datos sintéticos incompletos para patrón {pattern}"

                spectrum = synthetic_data['spectrum']
                if len(spectrum.wavelength) != len(wavelength):
                    return False, f"Longitud incorrecta para patrón {pattern}"

            return True, "Generación de audio sintético exitosa"

        except Exception as e:
            return False, f"Error generando audio sintético: {e}"

    def _test_parameter_mapping(self) -> Tuple[bool, str]:
        """Prueba el mapeo de parámetros de audio."""
        try:
            # Crear espectro de prueba
            wavelength = np.linspace(5000, 7000, 50)
            flux = np.random.normal(1.0, 0.1, 50)

            from ..domain.models.spectrum_data import SpectrumData
            spectrum = SpectrumData(wavelength=wavelength, flux=flux)

            # Calcular parámetros
            params = self.service.calculate_audio_parameters(spectrum)

            required_keys = ['frequencies', 'amplitudes', 'statistics']
            for key in required_keys:
                if key not in params:
                    return False, f"Parámetro requerido {key} no encontrado"

            return True, "Mapeo de parámetros exitoso"

        except Exception as e:
            return False, f"Error en mapeo de parámetros: {e}"

    def _test_status_reporting(self) -> Tuple[bool, str]:
        """Prueba el reporte de estado."""
        try:
            status = self.service.get_sonification_status()

            if not isinstance(status, SonificationStatus):
                return False, "Estado no es instancia de SonificationStatus"

            required_attrs = ['active', 'flux_sensitive', 'port', 'directory']
            for attr in required_attrs:
                if not hasattr(status, attr):
                    return False, f"Atributo requerido {attr} no encontrado"

            return True, "Reporte de estado exitoso"

        except Exception as e:
            return False, f"Error en reporte de estado: {e}"

    def _test_flux_sensitivity_toggle(self) -> Tuple[bool, str]:
        """Prueba el toggle de sensibilidad al flujo."""
        try:
            # Obtener estado inicial
            initial_state = self.service.get_flux_sensitivity()

            # Alternar
            new_state = self.service.toggle_flux_sensitivity()

            # Verificar cambio
            if new_state == initial_state:
                return False, "Toggle no cambió el estado"

            # Verificar consistencia
            current_state = self.service.get_flux_sensitivity()
            if current_state != new_state:
                return False, "Estado inconsistente después del toggle"

            return True, "Toggle de sensibilidad al flujo exitoso"

        except Exception as e:
            return False, f"Error en toggle: {e}"

    def _test_cleanup(self) -> Tuple[bool, str]:
        """Prueba la limpieza del servicio."""
        try:
            # Configurar estado inicial
            self.service.set_flux_sensitivity(True)

            # Ejecutar limpieza
            self.service.cleanup()

            # Verificar limpieza
            status = self.service.get_sonification_status()
            if status.active:
                return False, "Servicio sigue activo después de limpieza"

            return True, "Limpieza exitosa"

        except Exception as e:
            return False, f"Error en limpieza: {e}"