"""
Servicio de sonificación para ViewCube.

Este módulo maneja toda la funcionalidad de sonificación del sistema,
incluyendo la conversión de datos espectrales a audio y la integración
con sistemas de sonificación externos.
"""

import os
import sys
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
import threading
import time

from ..domain.models.spectrum_data import SpectrumData
from ..domain.models.cube_data import CubeData
from ..domain.entities.astronomical_entities import AstronomicalObject


class SonificationService:
    """
    Servicio de sonificación para ViewCube.

    Gestiona la conversión de datos espectrales a audio y la integración
    con sistemas de sonificación como SoniCube.
    """

    def __init__(self,
                 sonification_directory: Optional[str] = None,
                 auto_start: bool = False,
                 default_port: int = 9970):
        """
        Inicializa el servicio de sonificación.

        Args:
            sonification_directory: Directorio base para sonificación
            auto_start: Iniciar automáticamente el motor de sonificación
            default_port: Puerto por defecto para comunicación OSC
        """
        self.sonification_directory = sonification_directory
        self.default_port = default_port
        self.auto_start = auto_start

        # Estado del servicio
        self._active = False
        self._sonicube_instance = None
        self._current_cube_data = None
        self._current_spectrum = None
        self._flux_sensitive = True
        self._sonification_thread = None

        # Configuración de audio
        self._audio_config = {
            'sample_rate': 48000,
            'buffer_size': 1024,
            'channels': 2,  # Estéreo para espacialización
            'bit_depth': 16
        }

        # Configuración de sonificación
        self._sonification_config = {
            'frequency_range': (200, 2000),  # Hz
            'volume_range': (0.0, 1.0),
            'spatial_range': 360,  # grados
            'latent_dimensions': 6,
            'a_weighting': True,
            'flux_sensitivity': True,
            'distance_attenuation': True
        }

        # Cache de datos procesados
        self._processed_data_cache = {}

        # Inicializar si está configurado
        if auto_start and sonification_directory:
            self.initialize_sonification()

    def initialize_sonification(self) -> bool:
        """
        Inicializa el sistema de sonificación.

        Returns:
            True si la inicialización fue exitosa
        """
        if not self.sonification_directory:
            print(">>> Directorio de sonificación no especificado")
            return False

        if not os.path.exists(self.sonification_directory):
            print(f">>> Directorio de sonificación no existe: {self.sonification_directory}")
            return False

        try:
            # Intentar importar e inicializar SoniCube
            self._initialize_sonicube()
            self._active = True
            print(">>> Sistema de sonificación inicializado correctamente")
            return True

        except Exception as e:
            print(f">>> Error inicializando sonificación: {e}")
            return False

    def _initialize_sonicube(self) -> None:
        """Inicializa la instancia de SoniCube."""
        try:
            # Añadir el directorio de sonificación al path
            if self.sonification_directory not in sys.path:
                sys.path.append(self.sonification_directory)

            # Importar SoniCube
            from sonicube import SoniCube

            # Crear instancia (será inicializada cuando se carguen datos)
            self._sonicube_class = SoniCube
            print(">>> SoniCube importado correctamente")

        except ImportError as e:
            print(f">>> No se pudo importar SoniCube: {e}")
            raise
        except Exception as e:
            print(f">>> Error inicializando SoniCube: {e}")
            raise

    def is_active(self) -> bool:
        """
        Verifica si el sistema de sonificación está activo.

        Returns:
            True si está activo
        """
        return self._active

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
        if not self._active:
            print(">>> Sistema de sonificación no está activo")
            return False

        try:
            # Almacenar datos del cubo
            self._current_cube_data = cube_data

            # Crear instancia de SoniCube
            self._sonicube_instance = self._sonicube_class(
                fig=matplotlib_figure,
                file=fits_file,
                base_dir=self.sonification_directory,
                port=self.default_port,
                ref=reference_position,
                data=cube_data.data,
                flux_sensitive=self._flux_sensitive
            )

            print(">>> Sonificación de cubo configurada correctamente")
            return True

        except Exception as e:
            print(f">>> Error configurando sonificación de cubo: {e}")
            return False

    def sonify_spaxel(self, x: int, y: int, spectrum: Optional[SpectrumData] = None) -> bool:
        """
        Sonifica un spaxel específico.

        Args:
            x: Coordenada X del spaxel
            y: Coordenada Y del spaxel
            spectrum: Datos del espectro (opcional, se extraerá del cubo si no se proporciona)

        Returns:
            True si la sonificación fue exitosa
        """
        if not self._active or not self._sonicube_instance:
            return False

        try:
            # Obtener espectro si no se proporciona
            if spectrum is None and self._current_cube_data:
                spectrum = self._current_cube_data.get_spaxel_spectrum(x, y)

            if spectrum is None:
                return False

            # Sonificar usando SoniCube
            self._sonicube_instance.sonify(y, x, spectrum.flux)
            return True

        except Exception as e:
            print(f">>> Error en sonificación de spaxel: {e}")
            return False

    def sonify_spectrum(self, spectrum: SpectrumData,
                        position: Optional[Tuple[int, int]] = None) -> bool:
        """
        Sonifica un espectro específico.

        Args:
            spectrum: Datos del espectro
            position: Posición espacial del espectro (opcional)

        Returns:
            True si la sonificación fue exitosa
        """
        if not self._active or not self._sonicube_instance:
            return False

        try:
            # Usar posición por defecto si no se especifica
            if position is None:
                position = (0, 0)

            y, x = position
            self._sonicube_instance.sonify(y, x, spectrum.flux)
            return True

        except Exception as e:
            print(f">>> Error en sonificación de espectro: {e}")
            return False

    def toggle_flux_sensitivity(self) -> bool:
        """
        Alterna la sensibilidad al flujo en la sonificación.

        Returns:
            Nuevo estado de sensibilidad al flujo
        """
        self._flux_sensitive = not self._flux_sensitive

        if self._sonicube_instance:
            self._sonicube_instance.flux_sensitive = self._flux_sensitive

        print(f">>> Sensibilidad al flujo: {'Activada' if self._flux_sensitive else 'Desactivada'}")
        return self._flux_sensitive

    def set_flux_sensitivity(self, enabled: bool) -> None:
        """
        Establece la sensibilidad al flujo.

        Args:
            enabled: True para activar, False para desactivar
        """
        self._flux_sensitive = enabled

        if self._sonicube_instance:
            self._sonicube_instance.flux_sensitive = enabled

    def get_flux_sensitivity(self) -> bool:
        """
        Obtiene el estado actual de sensibilidad al flujo.

        Returns:
            True si está activada
        """
        return self._flux_sensitive

    def stop_sonification(self) -> None:
        """Detiene toda la sonificación activa."""
        if self._sonicube_instance:
            try:
                self._sonicube_instance.stop_sound()
            except Exception as e:
                print(f">>> Error deteniendo sonificación: {e}")

    def start_continuous_sonification(self,
                                      spectrum_generator,
                                      position_generator=None,
                                      interval: float = 0.1) -> None:
        """
        Inicia sonificación continua basada en generadores.

        Args:
            spectrum_generator: Generador que produce datos espectrales
            position_generator: Generador que produce posiciones (opcional)
            interval: Intervalo entre sonificaciones en segundos
        """
        if self._sonification_thread and self._sonification_thread.is_alive():
            self.stop_continuous_sonification()

        self._continuous_active = True
        self._sonification_thread = threading.Thread(
            target=self._continuous_sonification_loop,
            args=(spectrum_generator, position_generator, interval)
        )
        self._sonification_thread.daemon = True
        self._sonification_thread.start()

    def stop_continuous_sonification(self) -> None:
        """Detiene la sonificación continua."""
        self._continuous_active = False
        if self._sonification_thread:
            self._sonification_thread.join(timeout=1.0)
        self.stop_sonification()

    def _continuous_sonification_loop(self,
                                      spectrum_generator,
                                      position_generator,
                                      interval: float) -> None:
        """Loop interno para sonificación continua."""
        while self._continuous_active:
            try:
                # Obtener siguiente espectro
                spectrum = next(spectrum_generator, None)
                if spectrum is None:
                    break

                # Obtener posición si hay generador
                position = None
                if position_generator:
                    position = next(position_generator, None)

                # Sonificar
                self.sonify_spectrum(spectrum, position)

                # Esperar intervalo
                time.sleep(interval)

            except StopIteration:
                break
            except Exception as e:
                print(f">>> Error en sonificación continua: {e}")
                break

    def create_audio_mapping(self, spectrum: SpectrumData) -> Dict[str, Any]:
        """
        Crea un mapeo de audio para un espectro.

        Args:
            spectrum: Datos del espectro

        Returns:
            Diccionario con parámetros de audio
        """
        if not self._active:
            return {}

        try:
            # Calcular estadísticas del espectro
            flux_stats = self._calculate_spectrum_statistics(spectrum)

            # Mapear a parámetros de audio
            audio_params = self._map_to_audio_parameters(spectrum, flux_stats)

            return audio_params

        except Exception as e:
            print(f">>> Error creando mapeo de audio: {e}")
            return {}

    def _calculate_spectrum_statistics(self, spectrum: SpectrumData) -> Dict[str, float]:
        """Calcula estadísticas del espectro para mapeo de audio."""
        valid_flux = spectrum.flux[np.isfinite(spectrum.flux)]

        if len(valid_flux) == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0
            }

        return {
            'mean': float(np.mean(valid_flux)),
            'median': float(np.median(valid_flux)),
            'std': float(np.std(valid_flux)),
            'min': float(np.min(valid_flux)),
            'max': float(np.max(valid_flux)),
            'range': float(np.max(valid_flux) - np.min(valid_flux))
        }

    def _map_to_audio_parameters(self,
                                 spectrum: SpectrumData,
                                 stats: Dict[str, float]) -> Dict[str, Any]:
        """Mapea estadísticas del espectro a parámetros de audio."""
        # Mapear frecuencias basado en longitud de onda
        freq_min, freq_max = self._sonification_config['frequency_range']

        if len(spectrum.wavelength) > 0:
            wave_min, wave_max = np.min(spectrum.wavelength), np.max(spectrum.wavelength)
            # Mapeo inverso: longitudes de onda más cortas -> frecuencias más altas
            frequencies = freq_min + (freq_max - freq_min) * (
                    (wave_max - spectrum.wavelength) / (wave_max - wave_min)
            )
        else:
            frequencies = np.linspace(freq_min, freq_max, len(spectrum.flux))

        # Mapear amplitudes basado en flujo
        vol_min, vol_max = self._sonification_config['volume_range']
        if stats['range'] > 0:
            amplitudes = vol_min + (vol_max - vol_min) * (
                    (spectrum.flux - stats['min']) / stats['range']
            )
        else:
            amplitudes = np.full_like(spectrum.flux, vol_min)

        # Aplicar sensibilidad al flujo
        if self._flux_sensitive:
            flux_factor = np.clip(stats['median'] / (stats['max'] + 1e-10), 0.1, 1.0)
            amplitudes *= flux_factor

        return {
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'duration': 1.0,  # Duración en segundos
            'sample_rate': self._audio_config['sample_rate'],
            'flux_statistics': stats
        }

    def process_astronomical_object(self,
                                    astronomical_object: AstronomicalObject,
                                    sonification_mode: str = 'spectrum') -> bool:
        """
        Procesa un objeto astronómico para sonificación.

        Args:
            astronomical_object: Objeto astronómico
            sonification_mode: Modo de sonificación ('spectrum', 'cube', 'integrated')

        Returns:
            True si el procesamiento fue exitoso
        """
        if not self._active:
            return False

        try:
            cube_data = astronomical_object.get_cube_data()
            if cube_data is None:
                print(">>> No hay datos de cubo en el objeto astronómico")
                return False

            if sonification_mode == 'spectrum':
                # Sonificar spaxel central
                center_x, center_y = cube_data.n_x // 2, cube_data.n_y // 2
                return self.sonify_spaxel(center_x, center_y)

            elif sonification_mode == 'integrated':
                # Sonificar espectro integrado
                integrated_spectrum = cube_data.get_mean_spectrum()
                return self.sonify_spectrum(integrated_spectrum, (center_y, center_x))

            elif sonification_mode == 'cube':
                # Configurar sonificación de cubo completo
                # (requiere archivo FITS original)
                print(">>> Modo de cubo requiere configuración adicional")
                return False

            return False

        except Exception as e:
            print(f">>> Error procesando objeto astronómico: {e}")
            return False

    def get_sonification_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema de sonificación.

        Returns:
            Diccionario con información de estado
        """
        status = {
            'active': self._active,
            'sonicube_initialized': self._sonicube_instance is not None,
            'flux_sensitive': self._flux_sensitive,
            'directory': self.sonification_directory,
            'port': self.default_port,
            'current_cube_loaded': self._current_cube_data is not None,
            'continuous_active': getattr(self, '_continuous_active', False)
        }

        if self._sonicube_instance:
            status.update({
                'sonicube_cube_loaded': getattr(self._sonicube_instance, 'is_cube', False),
                'sonicube_autoencoder_loaded': getattr(self._sonicube_instance, 'autoencoder', None) is not None
            })

        return status

    def configure_audio_settings(self, **kwargs) -> None:
        """
        Configura los parámetros de audio.

        Args:
            **kwargs: Parámetros de configuración de audio
        """
        self._audio_config.update(kwargs)

    def configure_sonification_settings(self, **kwargs) -> None:
        """
        Configura los parámetros de sonificación.

        Args:
            **kwargs: Parámetros de configuración de sonificación
        """
        self._sonification_config.update(kwargs)

        # Actualizar instancia de SoniCube si existe
        if self._sonicube_instance:
            if 'flux_sensitivity' in kwargs:
                self._sonicube_instance.flux_sensitive = kwargs['flux_sensitivity']

    def get_audio_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración actual de audio.

        Returns:
            Diccionario con configuración de audio
        """
        return self._audio_config.copy()

    def get_sonification_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración actual de sonificación.

        Returns:
            Diccionario con configuración de sonificación
        """
        return self._sonification_config.copy()

    def validate_sonification_data(self, file_path: str) -> bool:
        """
        Valida si un archivo puede ser sonificado.

        Args:
            file_path: Ruta al archivo FITS

        Returns:
            True si el archivo puede ser sonificado
        """
        if not self._active or not self._sonicube_instance:
            return False

        try:
            # Usar el método de validación de SoniCube
            return self._sonicube_instance.check_datacube(file_path)
        except:
            return False

    def create_synthetic_sonification(self,
                                      wavelength: np.ndarray,
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

        # Crear mapeo de audio
        audio_params = self.create_audio_mapping(synthetic_spectrum)

        return {
            'spectrum': synthetic_spectrum,
            'audio_params': audio_params,
            'pattern': flux_pattern,
            'duration': duration
        }

    def cleanup(self) -> None:
        """Limpia recursos y cierra conexiones de sonificación."""
        # Detener sonificación continua
        if hasattr(self, '_continuous_active'):
            self.stop_continuous_sonification()

        # Detener sonificación actual
        self.stop_sonification()

        # Cerrar SoniCube
        if self._sonicube_instance:
            try:
                if hasattr(self._sonicube_instance, 'close_sound'):
                    self._sonicube_instance.close_sound()
            except Exception as e:
                print(f">>> Error cerrando SoniCube: {e}")

        # Limpiar estado
        self._active = False
        self._sonicube_instance = None
        self._current_cube_data = None
        self._current_spectrum = None
        self._processed_data_cache.clear()

        print(">>> Servicio de sonificación limpiado")

    def __del__(self):
        """Destructor para limpiar recursos."""
        try:
            self.cleanup()
        except:
            pass