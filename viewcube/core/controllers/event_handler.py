"""Gestor centralizado de eventos de teclado y mouse."""

from typing import Callable, Dict, Any, Optional
import matplotlib.pyplot as plt


class EventHandler:
    """Gestor centralizado de eventos para la aplicación ViewCube."""

    def __init__(self, controller):
        """
        Inicializa el gestor de eventos.

        Parameters:
        -----------
        controller : CubeController
            Referencia al controlador principal
        """
        self.controller = controller
        self.key_bindings: Dict[str, Callable] = {}
        self.mouse_bindings: Dict[str, Callable] = {}
        self._setup_default_bindings()

    def _setup_default_bindings(self) -> None:
        """Configura los bindings por defecto."""
        # Eventos de teclado
        self.key_bindings = {
            't': self._next_filter,
            'T': self._previous_filter,
            '*': self._clear_selections,
            's': self._toggle_display_mode,
            'S': self._save_spectra,
            'q': self._quit_application,
            'w': self._window_manager,
            'l': self._lambda_limits,
            'Y': self._flux_limits,
            'E': self._xy_limits,
            'h': self._toggle_sonification,
            'r': self._redshift_mode,
            'R': self._rest_wavelength,
            'e': self._error_mode,
            'f': self._fit_spectrum,
            'i': self._info_mode,
            'z': self._zoom_mode,
            'b': self._toggle_selector,
            'B': self._toggle_selector,
            'v': self._synthetic_mode,
            'p': self._passband_mode,
            'c': self._continuum_mode,
            'n': self._normalize_mode,
            '0': self._reset_view,
            '+': self._zoom_in,
            '-': self._zoom_out
        }

        # Eventos de mouse
        self.mouse_bindings = {
            'button_press': self._on_mouse_press,
            'button_release': self._on_mouse_release,
            'motion_notify': self._on_mouse_motion,
            'pick': self._on_pick_event
        }

    def connect_events(self) -> None:
        """Conecta todos los eventos a las figuras."""
        # Conectar eventos de teclado
        self.controller.spaxel_viewer.figure.canvas.mpl_connect(
            "key_press_event", self._on_key_press
        )
        self.controller.spectrum_viewer.figure.canvas.mpl_connect(
            "key_press_event", self._on_key_press
        )

        # Conectar eventos de mouse en spaxel viewer
        self.controller.spaxel_viewer.figure.canvas.mpl_connect(
            "button_press_event", self._on_mouse_press
        )
        self.controller.spaxel_viewer.figure.canvas.mpl_connect(
            "button_release_event", self._on_mouse_release
        )
        self.controller.spaxel_viewer.figure.canvas.mpl_connect(
            "motion_notify_event", self._on_mouse_motion
        )

        # Conectar eventos de mouse en spectrum viewer
        self.controller.spectrum_viewer.figure.canvas.mpl_connect(
            "button_press_event", self._on_mouse_press
        )
        self.controller.spectrum_viewer.figure.canvas.mpl_connect(
            "pick_event", self._on_pick_event
        )

    def _on_key_press(self, event) -> None:
        """Maneja eventos de teclado."""
        if event.key in self.key_bindings:
            self.key_bindings[event.key](event)

    def _on_mouse_press(self, event) -> None:
        """Maneja eventos de click del mouse."""
        self.controller._on_mouse_press(event)

    def _on_mouse_release(self, event) -> None:
        """Maneja eventos de liberación del mouse."""
        self.controller._on_mouse_release(event)

    def _on_mouse_motion(self, event) -> None:
        """Maneja eventos de movimiento del mouse."""
        self.controller._on_mouse_motion(event)

    def _on_pick_event(self, event) -> None:
        """Maneja eventos de selección."""
        self.controller._on_pick_event(event)

    # Métodos de eventos específicos
    def _next_filter(self, event) -> None:
        """Cambia al siguiente filtro."""
        self.controller.filter_manager.next_filter()
        self.controller._update_spaxel_display()
        if self.controller.current_spaxel:
            self.controller._update_spectrum_display(self.controller.current_spaxel)

    def _previous_filter(self, event) -> None:
        """Cambia al filtro anterior."""
        self.controller.filter_manager.previous_filter()
        self.controller._update_spaxel_display()
        if self.controller.current_spaxel:
            self.controller._update_spectrum_display(self.controller.current_spaxel)

    def _clear_selections(self, event) -> None:
        """Limpia todas las selecciones."""
        self.controller.spaxel_viewer.clear_selections()
        self.controller.spectrum_viewer.clear_display()
        self.controller.selected_spaxels.clear()
        self.controller.current_spaxel = None

    def _toggle_display_mode(self, event) -> None:
        """Alterna el modo de visualización."""
        if len(self.controller.selected_spaxels) > 0:
            self.controller._show_multiple_spectra()

    def _save_spectra(self, event) -> None:
        """Guarda espectros seleccionados."""
        self.controller._save_selected_spectra()

    def _quit_application(self, event) -> None:
        """Cierra la aplicación."""
        import sys
        plt.close('all')
        sys.exit()

    def _window_manager(self, event) -> None:
        """Abre el gestor de ventanas."""
        self.controller.WindowManager()

    def _lambda_limits(self, event) -> None:
        """Configura límites de longitud de onda."""
        self.controller.LambdaLimits()

    def _flux_limits(self, event) -> None:
        """Configura límites de flujo."""
        self.controller.FluxLimits()

    def _xy_limits(self, event) -> None:
        """Configura límites espaciales."""
        self.controller.xyLimits()

    def _toggle_sonification(self, event) -> None:
        """Activa/desactiva sonificación."""
        if hasattr(self.controller, 'soni_mode'):
            self.controller.soni_mode = not self.controller.soni_mode
            if self.controller.soni_mode:
                self.controller._init_sonification()

    def _redshift_mode(self, event) -> None:
        """Activa modo redshift."""
        self.controller.Redshift(event)

    def _rest_wavelength(self, event) -> None:
        """Cambia a longitud de onda en reposo."""
        self.controller.RestWave(event)

    def _error_mode(self, event) -> None:
        """Activa visualización de errores."""
        self.controller.ErrorSpec(event)

    def _fit_spectrum(self, event) -> None:
        """Abre herramientas de ajuste espectral."""
        self.controller.FitSpec(event)

    def _info_mode(self, event) -> None:
        """Muestra información del espectro."""
        self.controller.GetSpectraInfo(event)

    def _zoom_mode(self, event) -> None:
        """Activa modo zoom."""
        pass  # Implementar según necesidades

    def _toggle_selector(self, event) -> None:
        """Activa/desactiva selector rectangular."""
        self.controller.toggle_selector(event)

    def _synthetic_mode(self, event) -> None:
        """Activa visualización de espectros sintéticos."""
        self.controller.SynthSpec(event)

    def _passband_mode(self, event) -> None:
        """Activa modo passband."""
        pass  # Implementar según necesidades

    def _continuum_mode(self, event) -> None:
        """Activa modo continuum."""
        pass  # Implementar según necesidades

    def _normalize_mode(self, event) -> None:
        """Activa normalización."""
        pass  # Implementar según necesidades

    def _reset_view(self, event) -> None:
        """Resetea la vista."""
        self.controller._reset_view()

    def _zoom_in(self, event) -> None:
        """Zoom in."""
        pass  # Implementar según necesidades

    def _zoom_out(self, event) -> None:
        """Zoom out."""
        pass  # Implementar según necesidades

    def add_key_binding(self, key: str, callback: Callable) -> None:
        """Añade un nuevo binding de teclado."""
        self.key_bindings[key] = callback

    def remove_key_binding(self, key: str) -> None:
        """Remueve un binding de teclado."""
        if key in self.key_bindings:
            del self.key_bindings[key]

    def add_mouse_binding(self, event_type: str, callback: Callable) -> None:
        """Añade un nuevo binding de mouse."""
        self.mouse_bindings[event_type] = callback

    def remove_mouse_binding(self, event_type: str) -> None:
        """Remueve un binding de mouse."""
        if event_type in self.mouse_bindings:
            del self.mouse_bindings[event_type]
