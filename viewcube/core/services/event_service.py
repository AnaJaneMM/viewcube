"""
Servicio de eventos para ViewCube.

Este módulo centraliza todo el manejo de eventos del sistema,
incluyendo eventos de teclado, mouse, interfaz y coordinación entre componentes.
"""

import sys
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.backend_bases as backend_bases

from ..domain.models.spectrum_data import SpectrumData
from ..domain.models.cube_data import CubeData
from ..domain.entities.astronomical_entities import AstronomicalObject, Spaxel


class EventType(Enum):
    """Tipos de eventos soportados por el sistema."""
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    MOUSE_RELEASE = "mouse_release"
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    SPAXEL_SELECT = "spaxel_select"
    SPECTRUM_UPDATE = "spectrum_update"
    FILTER_CHANGE = "filter_change"
    VIEW_UPDATE = "view_update"
    SONIFICATION = "sonification"
    SAVE_REQUEST = "save_request"
    WINDOW_MANAGER = "window_manager"
    LAMBDA_LIMITS = "lambda_limits"
    FLUX_LIMITS = "flux_limits"
    ERROR_TOGGLE = "error_toggle"
    FLAG_TOGGLE = "flag_toggle"
    REDSHIFT_CHANGE = "redshift_change"
    REST_WAVE_TOGGLE = "rest_wave_toggle"
    PASSBAND_INTERACTION = "passband_interaction"
    RECTANGLE_SELECT = "rectangle_select"
    COLORMAP_CHANGE = "colormap_change"
    SYNTHETIC_TOGGLE = "synthetic_toggle"
    FIT_SPECTRUM = "fit_spectrum"


@dataclass
class EventData:
    """Contenedor de datos para eventos."""
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: float
    coordinates: Optional[Tuple[float, float]] = None
    key: Optional[str] = None
    button: Optional[int] = None


class EventService:
    """
    Servicio central para el manejo de eventos en ViewCube.

    Gestiona todos los eventos del sistema, incluyendo interacciones de usuario,
    actualizaciones de vista y coordinación entre componentes.
    """

    def __init__(self):
        """Inicializa el servicio de eventos."""
        self._event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._matplotlib_connections: Dict[str, List[int]] = defaultdict(list)
        self._active_viewers = {}
        self._current_selection = []
        self._rectangle_selector = None
        self._passband_interaction = {
            'active': False,
            'start_pos': None,
            'current_pos': None
        }

        # Estado del sistema
        self._system_state = {
            'current_mode': 'spectrum',  # 'spectrum' o 'selection'
            'selected_spaxels': [],
            'current_filter': None,
            'lambda_limits': None,
            'flux_limits': None,
            'colormap_settings': {},
            'show_errors': False,
            'show_flags': False,
            'show_synthetic': False,
            'rest_wavelength_mode': False,
            'sonification_active': False
        }

    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Registra un manejador para un tipo de evento específico.

        Args:
            event_type: Tipo de evento
            handler: Función a ejecutar cuando ocurra el evento
        """
        self._event_handlers[event_type].append(handler)

    def unregister_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Desregistra un manejador de eventos.

        Args:
            event_type: Tipo de evento
            handler: Función a desregistrar
        """
        if handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)

    def emit_event(self, event_data: EventData) -> None:
        """
        Emite un evento a todos los manejadores registrados.

        Args:
            event_data: Datos del evento
        """
        handlers = self._event_handlers.get(event_data.event_type, [])
        for handler in handlers:
            try:
                handler(event_data)
            except Exception as e:
                print(f"Error en manejador de evento {event_data.event_type}: {e}")

    def setup_matplotlib_events(self, figure, figure_name: str) -> None:
        """
        Configura los eventos de matplotlib para una figura.

        Args:
            figure: Figura de matplotlib
            figure_name: Nombre identificador de la figura
        """
        # Conectar eventos básicos
        cid_key = figure.canvas.mpl_connect('key_press_event',
                                            lambda event: self._handle_key_press(event, figure_name))
        cid_click = figure.canvas.mpl_connect('button_press_event',
                                              lambda event: self._handle_mouse_click(event, figure_name))
        cid_move = figure.canvas.mpl_connect('motion_notify_event',
                                             lambda event: self._handle_mouse_move(event, figure_name))
        cid_release = figure.canvas.mpl_connect('button_release_event',
                                                lambda event: self._handle_mouse_release(event, figure_name))

        # Guardar IDs de conexión para desconectar después
        self._matplotlib_connections[figure_name].extend([cid_key, cid_click, cid_move, cid_release])

        # Evento de pick (para elementos seleccionables)
        if hasattr(figure.canvas, 'mpl_connect'):
            cid_pick = figure.canvas.mpl_connect('pick_event',
                                                 lambda event: self._handle_pick_event(event, figure_name))
            self._matplotlib_connections[figure_name].append(cid_pick)

    def disconnect_matplotlib_events(self, figure, figure_name: str) -> None:
        """
        Desconecta los eventos de matplotlib para una figura.

        Args:
            figure: Figura de matplotlib
            figure_name: Nombre identificador de la figura
        """
        for cid in self._matplotlib_connections.get(figure_name, []):
            try:
                figure.canvas.mpl_disconnect(cid)
            except:
                pass
        self._matplotlib_connections[figure_name] = []

    def _handle_key_press(self, event, figure_name: str) -> None:
        """Maneja eventos de presión de teclas."""
        if event.key is None:
            return

        import time
        event_data = EventData(
            event_type=EventType.KEY_PRESS,
            source=figure_name,
            data={'figure': figure_name},
            timestamp=time.time(),
            key=event.key
        )

        # Manejar teclas específicas
        self._process_key_event(event.key, figure_name, event_data)

    def _process_key_event(self, key: str, figure_name: str, event_data: EventData) -> None:
        """
        Procesa eventos de teclado específicos.

        Args:
            key: Tecla presionada
            figure_name: Nombre de la figura
            event_data: Datos del evento
        """
        # Teclas de navegación y selección
        if key == 's':
            self._toggle_selection_mode()
        elif key == '*':
            self._clear_all_selections()
        elif key == 'S' and len(self._current_selection) > 0:
            self._emit_save_request()

        # Teclas de visualización
        elif key == 'w':
            self._emit_window_manager()
        elif key == 'l':
            self._emit_lambda_limits()
        elif key == 'Y':
            self._emit_flux_limits()
        elif key == 'I':
            self._toggle_integrated_spectrum()

        # Teclas de filtros y wavelength
        elif key in ['t', 'T']:
            self._change_filter(key)
        elif key == 'a':
            self._toggle_filter_centering()
        elif key == 'c':
            self._toggle_continuum_removal()
        elif key == 'k':
            self._request_redshift_input()
        elif key == 'z':
            self._toggle_rest_wavelength()

        # Teclas de visualización avanzada
        elif key == 'e':
            self._toggle_error_display()
        elif key == 'f':
            self._toggle_flag_display()
        elif key == 'r':
            self._toggle_residuals()
        elif key == 'y':
            self._toggle_synthetic_spectrum()

        # Teclas de sonificación y ajuste
        elif key == 'h':
            self._toggle_sonification()
        elif key == 'j':
            self._toggle_flux_sensitivity()
        elif key == 'F':
            self._request_spectrum_fit()

        # Teclas de selección rectangular
        elif key == 'b':
            self._activate_rectangle_selector()
        elif key == 'B':
            self._deactivate_rectangle_selector()

        # Teclas de colormap
        elif key in ['+', '-', 'pageup', 'pagedown']:
            self._change_colormap(key)
        elif key == 'm':
            self._request_colormap_selection()
        elif key == 'i':
            self._invert_colormap()
        elif key in ['1', '2', '3', '4', '5']:
            self._change_color_scale(key)
        elif key == '0':
            self._show_available_scales()
        elif key == 'v':
            self._request_colormap_limits()

        # Salir
        elif key == 'q':
            self._request_quit()

        # Emitir evento
        self.emit_event(event_data)

    def _handle_mouse_click(self, event, figure_name: str) -> None:
        """Maneja eventos de clic del mouse."""
        if event.inaxes is None:
            return

        import time
        event_data = EventData(
            event_type=EventType.MOUSE_CLICK,
            source=figure_name,
            data={
                'figure': figure_name,
                'inaxes': event.inaxes,
                'x_data': event.xdata,
                'y_data': event.ydata
            },
            timestamp=time.time(),
            coordinates=(event.xdata, event.ydata) if event.xdata and event.ydata else None,
            button=event.button
        )

        # Procesar según el tipo de figura y modo
        if figure_name == 'spaxel_viewer':
            self._handle_spaxel_click(event_data)
        elif figure_name == 'spectrum_viewer':
            self._handle_spectrum_click(event_data)

        self.emit_event(event_data)

    def _handle_mouse_move(self, event, figure_name: str) -> None:
        """Maneja eventos de movimiento del mouse."""
        if event.inaxes is None:
            return

        import time
        event_data = EventData(
            event_type=EventType.MOUSE_MOVE,
            source=figure_name,
            data={
                'figure': figure_name,
                'inaxes': event.inaxes,
                'x_data': event.xdata,
                'y_data': event.ydata
            },
            timestamp=time.time(),
            coordinates=(event.xdata, event.ydata) if event.xdata and event.ydata else None,
            button=getattr(event, 'button', None)
        )

        # Procesar según el contexto
        if figure_name == 'spaxel_viewer' and self._system_state['current_mode'] == 'spectrum':
            self._handle_spaxel_hover(event_data)
        elif figure_name == 'spectrum_viewer':
            self._handle_spectrum_hover(event_data)

        # Manejar interacciones de passband
        if self._passband_interaction['active']:
            self._handle_passband_move(event_data)

        # Manejar cambio dinámico de colormap
        if hasattr(event, 'button') and event.button == 3:  # Botón derecho
            self._handle_dynamic_colormap_change(event_data)

        self.emit_event(event_data)

    def _handle_mouse_release(self, event, figure_name: str) -> None:
        """Maneja eventos de liberación del mouse."""
        import time
        event_data = EventData(
            event_type=EventType.MOUSE_RELEASE,
            source=figure_name,
            data={
                'figure': figure_name,
                'inaxes': event.inaxes,
                'x_data': event.xdata,
                'y_data': event.ydata
            },
            timestamp=time.time(),
            coordinates=(event.xdata, event.ydata) if event.xdata and event.ydata else None,
            button=event.button
        )

        # Finalizar interacciones
        if self._passband_interaction['active']:
            self._handle_passband_release(event_data)

        self.emit_event(event_data)

    def _handle_pick_event(self, event, figure_name: str) -> None:
        """Maneja eventos de selección (pick) de elementos."""
        import time
        event_data = EventData(
            event_type=EventType.SPAXEL_SELECT,
            source=figure_name,
            data={
                'figure': figure_name,
                'artist': event.artist,
                'mouseevent': event.mouseevent
            },
            timestamp=time.time()
        )

        self.emit_event(event_data)

    def _handle_spaxel_click(self, event_data: EventData) -> None:
        """Procesa clics en el visor de spaxels."""
        x_data = event_data.data.get('x_data')
        y_data = event_data.data.get('y_data')

        if x_data is not None and y_data is not None:
            # Convertir coordenadas a índices de spaxel
            spaxel_coords = self._convert_to_spaxel_coordinates(x_data, y_data)

            if self._system_state['current_mode'] == 'selection':
                self._add_spaxel_to_selection(spaxel_coords)
            else:
                self._select_single_spaxel(spaxel_coords)

    def _handle_spaxel_hover(self, event_data: EventData) -> None:
        """Procesa hover sobre spaxels."""
        x_data = event_data.data.get('x_data')
        y_data = event_data.data.get('y_data')

        if x_data is not None and y_data is not None:
            spaxel_coords = self._convert_to_spaxel_coordinates(x_data, y_data)
            self._update_spectrum_preview(spaxel_coords)

    def _handle_spectrum_click(self, event_data: EventData) -> None:
        """Procesa clics en el visor de espectros."""
        # Iniciar interacción de passband si está habilitada
        if event_data.button == 1:  # Botón izquierdo
            self._start_passband_interaction(event_data)

    def _handle_spectrum_hover(self, event_data: EventData) -> None:
        """Procesa hover sobre el espectro."""
        # Actualizar información de wavelength/flux en la barra de estado
        pass

    def _toggle_selection_mode(self) -> None:
        """Alterna entre modo espectro y modo selección."""
        current_mode = self._system_state['current_mode']
        new_mode = 'selection' if current_mode == 'spectrum' else 'spectrum'
        self._system_state['current_mode'] = new_mode

        event_data = EventData(
            event_type=EventType.VIEW_UPDATE,
            source='event_service',
            data={'mode_change': new_mode},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _clear_all_selections(self) -> None:
        """Limpia todas las selecciones actuales."""
        self._current_selection = []
        self._system_state['selected_spaxels'] = []

        event_data = EventData(
            event_type=EventType.VIEW_UPDATE,
            source='event_service',
            data={'clear_selections': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _emit_save_request(self) -> None:
        """Emite solicitud de guardado."""
        event_data = EventData(
            event_type=EventType.SAVE_REQUEST,
            source='event_service',
            data={'selected_spaxels': self._current_selection},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _emit_window_manager(self) -> None:
        """Emite solicitud del administrador de ventanas."""
        event_data = EventData(
            event_type=EventType.WINDOW_MANAGER,
            source='event_service',
            data={},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _emit_lambda_limits(self) -> None:
        """Emite solicitud de configuración de límites lambda."""
        event_data = EventData(
            event_type=EventType.LAMBDA_LIMITS,
            source='event_service',
            data={},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _emit_flux_limits(self) -> None:
        """Emite solicitud de configuración de límites de flujo."""
        event_data = EventData(
            event_type=EventType.FLUX_LIMITS,
            source='event_service',
            data={},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _change_filter(self, key: str) -> None:
        """Cambia el filtro actual."""
        direction = 1 if key == 't' else -1

        event_data = EventData(
            event_type=EventType.FILTER_CHANGE,
            source='event_service',
            data={'direction': direction},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _toggle_filter_centering(self) -> None:
        """Alterna el centrado del filtro."""
        event_data = EventData(
            event_type=EventType.FILTER_CHANGE,
            source='event_service',
            data={'toggle_centering': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _toggle_continuum_removal(self) -> None:
        """Alterna la remoción de continuo."""
        event_data = EventData(
            event_type=EventType.FILTER_CHANGE,
            source='event_service',
            data={'toggle_continuum_removal': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _request_redshift_input(self) -> None:
        """Solicita entrada de redshift."""
        event_data = EventData(
            event_type=EventType.REDSHIFT_CHANGE,
            source='event_service',
            data={'request_input': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _toggle_rest_wavelength(self) -> None:
        """Alterna entre wavelength observada y en reposo."""
        current_state = self._system_state['rest_wavelength_mode']
        self._system_state['rest_wavelength_mode'] = not current_state

        event_data = EventData(
            event_type=EventType.REST_WAVE_TOGGLE,
            source='event_service',
            data={'rest_mode': not current_state},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _toggle_error_display(self) -> None:
        """Alterna la visualización de errores."""
        current_state = self._system_state['show_errors']
        self._system_state['show_errors'] = not current_state

        event_data = EventData(
            event_type=EventType.ERROR_TOGGLE,
            source='event_service',
            data={'show_errors': not current_state},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _toggle_flag_display(self) -> None:
        """Alterna la visualización de flags."""
        current_state = self._system_state['show_flags']
        self._system_state['show_flags'] = not current_state

        event_data = EventData(
            event_type=EventType.FLAG_TOGGLE,
            source='event_service',
            data={'show_flags': not current_state},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _toggle_residuals(self) -> None:
        """Alterna la visualización de residuos."""
        event_data = EventData(
            event_type=EventType.VIEW_UPDATE,
            source='event_service',
            data={'toggle_residuals': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _toggle_synthetic_spectrum(self) -> None:
        """Alterna la visualización del espectro sintético."""
        current_state = self._system_state['show_synthetic']
        self._system_state['show_synthetic'] = not current_state

        event_data = EventData(
            event_type=EventType.SYNTHETIC_TOGGLE,
            source='event_service',
            data={'show_synthetic': not current_state},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _toggle_sonification(self) -> None:
        """Alterna la sonificación."""
        current_state = self._system_state['sonification_active']
        self._system_state['sonification_active'] = not current_state

        event_data = EventData(
            event_type=EventType.SONIFICATION,
            source='event_service',
            data={'active': not current_state},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _toggle_flux_sensitivity(self) -> None:
        """Alterna la sensibilidad de flujo en sonificación."""
        event_data = EventData(
            event_type=EventType.SONIFICATION,
            source='event_service',
            data={'toggle_flux_sensitivity': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _request_spectrum_fit(self) -> None:
        """Solicita ajuste de espectro."""
        event_data = EventData(
            event_type=EventType.FIT_SPECTRUM,
            source='event_service',
            data={'current_selection': self._current_selection},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _activate_rectangle_selector(self) -> None:
        """Activa el selector rectangular."""
        event_data = EventData(
            event_type=EventType.RECTANGLE_SELECT,
            source='event_service',
            data={'activate': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _deactivate_rectangle_selector(self) -> None:
        """Desactiva el selector rectangular."""
        event_data = EventData(
            event_type=EventType.RECTANGLE_SELECT,
            source='event_service',
            data={'activate': False},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _change_colormap(self, key: str) -> None:
        """Cambia el mapa de colores."""
        direction = 1 if key in ['+', 'pageup'] else -1

        event_data = EventData(
            event_type=EventType.COLORMAP_CHANGE,
            source='event_service',
            data={'direction': direction},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _request_colormap_selection(self) -> None:
        """Solicita selección manual de colormap."""
        event_data = EventData(
            event_type=EventType.COLORMAP_CHANGE,
            source='event_service',
            data={'request_selection': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _invert_colormap(self) -> None:
        """Invierte el colormap actual."""
        event_data = EventData(
            event_type=EventType.COLORMAP_CHANGE,
            source='event_service',
            data={'invert': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _change_color_scale(self, key: str) -> None:
        """Cambia la escala de colores."""
        scale_map = {
            '1': 'linear',
            '2': 'ilog',
            '3': 'sqrt',
            '4': 'power',
            '5': 'asinh'
        }
        scale = scale_map.get(key)

        if scale:
            event_data = EventData(
                event_type=EventType.COLORMAP_CHANGE,
                source='event_service',
                data={'scale': scale},
                timestamp=time.time()
            )
            self.emit_event(event_data)

    def _show_available_scales(self) -> None:
        """Muestra las escalas disponibles."""
        event_data = EventData(
            event_type=EventType.COLORMAP_CHANGE,
            source='event_service',
            data={'show_scales': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _request_colormap_limits(self) -> None:
        """Solicita configuración de límites de colormap."""
        event_data = EventData(
            event_type=EventType.COLORMAP_CHANGE,
            source='event_service',
            data={'request_limits': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _request_quit(self) -> None:
        """Solicita salir de la aplicación."""
        event_data = EventData(
            event_type=EventType.VIEW_UPDATE,
            source='event_service',
            data={'quit_request': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _convert_to_spaxel_coordinates(self, x_data: float, y_data: float) -> Tuple[int, int]:
        """
        Convierte coordenadas de datos a coordenadas de spaxel.

        Args:
            x_data: Coordenada X en datos
            y_data: Coordenada Y en datos

        Returns:
            Tupla (x, y) de coordenadas de spaxel
        """
        # Esta conversión debe ser implementada según el tipo de datos
        # Para cubos: conversión directa a índices
        # Para RSS: conversión usando tabla de posiciones
        return (int(round(x_data)), int(round(y_data)))

    def _add_spaxel_to_selection(self, coords: Tuple[int, int]) -> None:
        """Añade un spaxel a la selección actual."""
        if coords not in self._current_selection:
            self._current_selection.append(coords)
            self._system_state['selected_spaxels'] = self._current_selection

            event_data = EventData(
                event_type=EventType.SPAXEL_SELECT,
                source='event_service',
                data={'coordinates': coords, 'selection_mode': True},
                timestamp=time.time()
            )
            self.emit_event(event_data)

    def _select_single_spaxel(self, coords: Tuple[int, int]) -> None:
        """Selecciona un único spaxel."""
        self._current_selection = [coords]
        self._system_state['selected_spaxels'] = self._current_selection

        event_data = EventData(
            event_type=EventType.SPAXEL_SELECT,
            source='event_service',
            data={'coordinates': coords, 'selection_mode': False},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _update_spectrum_preview(self, coords: Tuple[int, int]) -> None:
        """Actualiza la previsualización del espectro."""
        event_data = EventData(
            event_type=EventType.SPECTRUM_UPDATE,
            source='event_service',
            data={'coordinates': coords, 'preview': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def _start_passband_interaction(self, event_data: EventData) -> None:
        """Inicia la interacción con passband."""
        self._passband_interaction['active'] = True
        self._passband_interaction['start_pos'] = event_data.coordinates

        passband_event = EventData(
            event_type=EventType.PASSBAND_INTERACTION,
            source='event_service',
            data={'action': 'start', 'position': event_data.coordinates},
            timestamp=event_data.timestamp
        )
        self.emit_event(passband_event)

    def _handle_passband_move(self, event_data: EventData) -> None:
        """Maneja el movimiento durante interacción de passband."""
        self._passband_interaction['current_pos'] = event_data.coordinates

        passband_event = EventData(
            event_type=EventType.PASSBAND_INTERACTION,
            source='event_service',
            data={'action': 'move', 'position': event_data.coordinates},
            timestamp=event_data.timestamp
        )
        self.emit_event(passband_event)

    def _handle_passband_release(self, event_data: EventData) -> None:
        """Maneja la liberación durante interacción de passband."""
        if self._passband_interaction['active']:
            self._passband_interaction['active'] = False

            passband_event = EventData(
                event_type=EventType.PASSBAND_INTERACTION,
                source='event_service',
                data={'action': 'end', 'position': event_data.coordinates},
                timestamp=event_data.timestamp
            )
            self.emit_event(passband_event)

    def _handle_dynamic_colormap_change(self, event_data: EventData) -> None:
        """Maneja el cambio dinámico de colormap con botón derecho."""
        x_data = event_data.data.get('x_data')
        y_data = event_data.data.get('y_data')

        if x_data is not None and y_data is not None:
            colormap_event = EventData(
                event_type=EventType.COLORMAP_CHANGE,
                source='event_service',
                data={
                    'dynamic_change': True,
                    'x_position': x_data,
                    'y_position': y_data
                },
                timestamp=event_data.timestamp
            )
            self.emit_event(colormap_event)

    def _toggle_integrated_spectrum(self) -> None:
        """Alterna la visualización del espectro integrado."""
        event_data = EventData(
            event_type=EventType.SPECTRUM_UPDATE,
            source='event_service',
            data={'toggle_integrated': True},
            timestamp=time.time()
        )
        self.emit_event(event_data)

    def get_system_state(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema.

        Returns:
            Diccionario con el estado del sistema
        """
        return self._system_state.copy()

    def update_system_state(self, **kwargs) -> None:
        """
        Actualiza el estado del sistema.

        Args:
            **kwargs: Valores a actualizar en el estado
        """
        self._system_state.update(kwargs)

    def get_current_selection(self) -> List[Tuple[int, int]]:
        """
        Obtiene la selección actual de spaxels.

        Returns:
            Lista de coordenadas de spaxels seleccionados
        """
        return self._current_selection.copy()

    def set_current_selection(self, selection: List[Tuple[int, int]]) -> None:
        """
        Establece la selección actual de spaxels.

        Args:
            selection: Lista de coordenadas de spaxels
        """
        self._current_selection = selection
        self._system_state['selected_spaxels'] = selection

    def cleanup(self) -> None:
        """Limpia recursos y desconecta eventos."""
        # Desconectar todos los eventos de matplotlib
        for figure_name in list(self._matplotlib_connections.keys()):
            # Nota: necesitaríamos la referencia a la figura para desconectar
            self._matplotlib_connections[figure_name] = []

        # Limpiar manejadores
        self._event_handlers.clear()

        # Resetear estado
        self._current_selection = []
        self._system_state = {
            'current_mode': 'spectrum',
            'selected_spaxels': [],
            'current_filter': None,
            'lambda_limits': None,
            'flux_limits': None,
            'colormap_settings': {},
            'show_errors': False,
            'show_flags': False,
            'show_synthetic': False,
            'rest_wavelength_mode': False,
            'sonification_active': False
        }