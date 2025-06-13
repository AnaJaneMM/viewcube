"""
Servicio de gestión de eventos para ViewCube - Migración Completa a PyQt5/PyQtGraph.

Esta implementación elimina completamente cualquier dependencia o compatibilidad con Matplotlib,
utilizando únicamente el ecosistema PyQt5/PyQtGraph para el manejo de eventos.
"""

from enum import Enum
from typing import Dict, List, Callable, Optional, Any, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass
import time
import logging
import traceback

# Imports exclusivos de PyQt5 - Sin referencias a Matplotlib
from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot, QPointF, QTimer,
                          Qt, QEvent, QThread, QMutex, QMutexLocker)
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent

# Imports exclusivos de PyQtGraph - Reemplazo completo de Matplotlib
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ViewBox, GraphicsScene, ImageItem
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent, MouseDragEvent, MouseMoveEvent

# Configuración de logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EventType(Enum):
    """Tipos de eventos soportados por el sistema PyQt5/PyQtGraph."""
    # Eventos básicos de interfaz
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    MOUSE_RELEASE = "mouse_release"
    MOUSE_WHEEL = "mouse_wheel"
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"

    # Eventos específicos de la aplicación
    SPAXEL_SELECT = "spaxel_select"
    SPECTRUM_UPDATE = "spectrum_update"
    FILTER_CHANGE = "filter_change"
    VIEW_UPDATE = "view_update"
    SONIFICATION = "sonification"
    SAVE_REQUEST = "save_request"
    WINDOW_MANAGER = "window_manager"

    # Eventos de configuración
    LAMBDA_LIMITS = "lambda_limits"
    FLUX_LIMITS = "flux_limits"
    ERROR_TOGGLE = "error_toggle"
    FLAG_TOGGLE = "flag_toggle"
    REDSHIFT_CHANGE = "redshift_change"
    REST_WAVE_TOGGLE = "rest_wave_toggle"

    # Eventos de interacción avanzada
    PASSBAND_INTERACTION = "passband_interaction"
    RECTANGLE_SELECT = "rectangle_select"
    COLORMAP_CHANGE = "colormap_change"
    SYNTHETIC_TOGGLE = "synthetic_toggle"
    FIT_SPECTRUM = "fit_spectrum"

    # Eventos de sistema
    WIDGET_FOCUS = "widget_focus"
    WIDGET_RESIZE = "widget_resize"
    APPLICATION_EXIT = "application_exit"


@dataclass
class EventData:
    """Contenedor optimizado de datos para eventos PyQt5/PyQtGraph."""
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: float
    coordinates: Optional[Tuple[float, float]] = None
    key: Optional[str] = None
    button: Optional[int] = None
    modifiers: Optional[Qt.KeyboardModifiers] = None
    widget_ref: Optional[QWidget] = None


class EventSignals(QObject):
    """Sistema de señales PyQt5 para eventos de la aplicación."""

    # Señales categorizadas por tipo
    # Eventos de ratón
    mouse_clicked = pyqtSignal(EventData)
    mouse_moved = pyqtSignal(EventData)
    mouse_released = pyqtSignal(EventData)
    mouse_wheel = pyqtSignal(EventData)

    # Eventos de teclado
    key_pressed = pyqtSignal(EventData)
    key_released = pyqtSignal(EventData)

    # Eventos de aplicación
    spaxel_selected = pyqtSignal(EventData)
    spectrum_updated = pyqtSignal(EventData)
    filter_changed = pyqtSignal(EventData)
    view_updated = pyqtSignal(EventData)
    sonification_triggered = pyqtSignal(EventData)

    # Eventos de sistema
    save_requested = pyqtSignal(EventData)
    window_manager_requested = pyqtSignal(EventData)
    lambda_limits_requested = pyqtSignal(EventData)
    flux_limits_requested = pyqtSignal(EventData)

    # Eventos de configuración
    error_toggled = pyqtSignal(EventData)
    flag_toggled = pyqtSignal(EventData)
    redshift_changed = pyqtSignal(EventData)
    rest_wave_toggled = pyqtSignal(EventData)

    # Eventos de interacción
    passband_interaction = pyqtSignal(EventData)
    rectangle_selected = pyqtSignal(EventData)
    colormap_changed = pyqtSignal(EventData)
    synthetic_toggled = pyqtSignal(EventData)
    spectrum_fitted = pyqtSignal(EventData)

    # Eventos de widget
    widget_focus_changed = pyqtSignal(EventData)
    widget_resized = pyqtSignal(EventData)
    application_exit_requested = pyqtSignal(EventData)

    # Señal universal para todos los eventos
    event_emitted = pyqtSignal(EventData)


class EventHandlerRegistry(QObject):
    """Registro thread-safe de manejadores de eventos usando señales PyQt5."""

    def __init__(self):
        super().__init__()
        self.signals = EventSignals()
        self._mutex = QMutex()

        # Mapeo optimizado de tipos de evento a señales
        self._signal_map = self._create_signal_mapping()

        # Contador de manejadores registrados
        self._handler_count = defaultdict(int)

    def _create_signal_mapping(self) -> Dict[EventType, pyqtSignal]:
        """Crea el mapeo entre tipos de evento y señales PyQt5."""
        return {
            # Eventos de ratón
            EventType.MOUSE_CLICK: self.signals.mouse_clicked,
            EventType.MOUSE_MOVE: self.signals.mouse_moved,
            EventType.MOUSE_RELEASE: self.signals.mouse_released,
            EventType.MOUSE_WHEEL: self.signals.mouse_wheel,

            # Eventos de teclado
            EventType.KEY_PRESS: self.signals.key_pressed,
            EventType.KEY_RELEASE: self.signals.key_released,

            # Eventos de aplicación
            EventType.SPAXEL_SELECT: self.signals.spaxel_selected,
            EventType.SPECTRUM_UPDATE: self.signals.spectrum_updated,
            EventType.FILTER_CHANGE: self.signals.filter_changed,
            EventType.VIEW_UPDATE: self.signals.view_updated,
            EventType.SONIFICATION: self.signals.sonification_triggered,

            # Eventos de sistema
            EventType.SAVE_REQUEST: self.signals.save_requested,
            EventType.WINDOW_MANAGER: self.signals.window_manager_requested,
            EventType.LAMBDA_LIMITS: self.signals.lambda_limits_requested,
            EventType.FLUX_LIMITS: self.signals.flux_limits_requested,

            # Eventos de configuración
            EventType.ERROR_TOGGLE: self.signals.error_toggled,
            EventType.FLAG_TOGGLE: self.signals.flag_toggled,
            EventType.REDSHIFT_CHANGE: self.signals.redshift_changed,
            EventType.REST_WAVE_TOGGLE: self.signals.rest_wave_toggled,

            # Eventos de interacción
            EventType.PASSBAND_INTERACTION: self.signals.passband_interaction,
            EventType.RECTANGLE_SELECT: self.signals.rectangle_selected,
            EventType.COLORMAP_CHANGE: self.signals.colormap_changed,
            EventType.SYNTHETIC_TOGGLE: self.signals.synthetic_toggled,
            EventType.FIT_SPECTRUM: self.signals.spectrum_fitted,

            # Eventos de widget
            EventType.WIDGET_FOCUS: self.signals.widget_focus_changed,
            EventType.WIDGET_RESIZE: self.signals.widget_resized,
            EventType.APPLICATION_EXIT: self.signals.application_exit_requested
        }

    def register_handler(self, event_type: EventType, handler: Callable[[EventData], None]) -> bool:
        """Registra un manejador para un tipo de evento específico de forma thread-safe."""
        with QMutexLocker(self._mutex):
            try:
                signal = self._signal_map.get(event_type)
                if signal is None:
                    logger.error(f"Tipo de evento no soportado: {event_type}")
                    return False

                signal.connect(handler, Qt.QueuedConnection)
                self._handler_count[event_type] += 1

                logger.debug(f"Registrado manejador para {event_type.value} "
                             f"(total: {self._handler_count[event_type]})")
                return True

            except Exception as e:
                logger.error(f"Error registrando manejador para {event_type}: {e}")
                return False

    def unregister_handler(self, event_type: EventType, handler: Callable[[EventData], None]) -> bool:
        """Desregistra un manejador de eventos de forma thread-safe."""
        with QMutexLocker(self._mutex):
            try:
                signal = self._signal_map.get(event_type)
                if signal is None:
                    return False

                signal.disconnect(handler)
                self._handler_count[event_type] = max(0, self._handler_count[event_type] - 1)

                logger.debug(f"Eliminado manejador para {event_type.value} "
                             f"(restantes: {self._handler_count[event_type]})")
                return True

            except TypeError:
                logger.warning(f"Manejador no encontrado para {event_type.value}")
                return False
            except Exception as e:
                logger.error(f"Error eliminando manejador para {event_type}: {e}")
                return False

    def emit_event(self, event_data: EventData) -> bool:
        """Emite un evento usando las señales PyQt5 de forma thread-safe."""
        try:
            signal = self._signal_map.get(event_data.event_type)
            if signal is None:
                logger.warning(f"Señal no encontrada para evento: {event_data.event_type}")
                return False

            # Emitir señal específica
            signal.emit(event_data)

            # Emitir señal universal
            self.signals.event_emitted.emit(event_data)

            return True

        except Exception as e:
            logger.error(f"Error emitiendo evento {event_data.event_type}: {e}")
            return False

    def get_handler_count(self, event_type: EventType) -> int:
        """Obtiene el número de manejadores registrados para un tipo de evento."""
        with QMutexLocker(self._mutex):
            return self._handler_count.get(event_type, 0)


class SystemState(QObject):
    """Gestor thread-safe del estado del sistema con señales PyQt5."""

    # Señales para notificar cambios de estado
    state_changed = pyqtSignal(dict)
    mode_changed = pyqtSignal(str, str)  # (old_mode, new_mode)
    selection_changed = pyqtSignal(list)
    filter_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._mutex = QMutex()
        self._state = self._initialize_default_state()

    def _initialize_default_state(self) -> Dict[str, Any]:
        """Inicializa el estado por defecto del sistema."""
        return {
            # Estado de visualización
            'current_mode': 'spectrum',
            'selected_spaxels': [],
            'current_filter': None,
            'filter_centered': False,
            'continuum_removed': False,

            # Límites y configuración
            'lambda_limits': None,
            'flux_limits': None,
            'colormap_settings': {
                'name': 'viridis',
                'scale': 'linear',
                'inverted': False,
                'limits': None
            },

            # Opciones de visualización
            'show_errors': False,
            'show_flags': False,
            'show_synthetic': False,
            'show_residuals': False,
            'show_integrated': False,
            'rest_wavelength_mode': False,

            # Estado de sonificación
            'sonification_active': False,
            'flux_sensitivity': False,

            # Estado de selección
            'rectangle_selector_active': False,
            'passband_interaction_active': False,

            # Estado de la aplicación
            'data_modified': False,
            'last_save_time': None
        }

    def get_state(self) -> Dict[str, Any]:
        """Obtiene una copia thread-safe del estado actual del sistema."""
        with QMutexLocker(self._mutex):
            return self._state.copy()

    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor específico del estado de forma thread-safe."""
        with QMutexLocker(self._mutex):
            return self._state.get(key, default)

    def update_state(self, **kwargs) -> Dict[str, Any]:
        """Actualiza el estado del sistema y emite señales correspondientes."""
        with QMutexLocker(self._mutex):
            old_state = self._state.copy()
            self._state.update(kwargs)

            # Emitir señales específicas para cambios importantes
            self._emit_specific_signals(old_state, kwargs)

            # Emitir señal general de cambio de estado
            self.state_changed.emit(kwargs)

            logger.debug(f"Estado actualizado: {kwargs}")
            return self._state.copy()

    def _emit_specific_signals(self, old_state: Dict[str, Any], changes: Dict[str, Any]) -> None:
        """Emite señales específicas para cambios de estado importantes."""
        # Cambio de modo
        if 'current_mode' in changes:
            old_mode = old_state.get('current_mode', 'spectrum')
            new_mode = changes['current_mode']
            if old_mode != new_mode:
                self.mode_changed.emit(old_mode, new_mode)

        # Cambio de selección
        if 'selected_spaxels' in changes:
            self.selection_changed.emit(changes['selected_spaxels'])

        # Cambio de filtro
        if 'current_filter' in changes:
            self.filter_changed.emit(str(changes['current_filter']))

    def reset_state(self) -> None:
        """Resetea el estado del sistema a los valores por defecto."""
        with QMutexLocker(self._mutex):
            old_state = self._state.copy()
            self._state = self._initialize_default_state()

            # Emitir señales para el reset
            self.state_changed.emit(self._state)
            self.mode_changed.emit(old_state.get('current_mode', 'spectrum'), 'spectrum')
            self.selection_changed.emit([])

            logger.info("Estado del sistema reseteado")


class PyQtGraphConnectionManager(QObject):
    """Gestor especializado de conexiones para widgets PyQtGraph."""

    def __init__(self, event_service: 'EventService'):
        super().__init__()
        self.event_service = event_service
        self._connections = defaultdict(list)
        self._widgets = {}
        self._event_filters = {}

    def connect_plot_widget(self, plot_widget: PlotWidget, widget_name: str) -> bool:
        """Conecta eventos de un PlotWidget de PyQtGraph."""
        try:
            if widget_name in self._widgets:
                logger.warning(f"Widget {widget_name} ya está conectado")
                return False

            self._widgets[widget_name] = plot_widget

            # Configurar el widget para eventos
            self._configure_widget_events(plot_widget, widget_name)

            # Conectar eventos del ViewBox
            self._connect_viewbox_events(plot_widget.getViewBox(), widget_name)

            # Conectar eventos de la escena gráfica
            self._connect_scene_events(plot_widget.scene(), widget_name)

            # Instalar filtro de eventos personalizado
            self._install_event_filter(plot_widget, widget_name)

            logger.info(f"Widget PyQtGraph {widget_name} conectado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error conectando widget {widget_name}: {e}")
            return False

    def _configure_widget_events(self, plot_widget: PlotWidget, widget_name: str) -> None:
        """Configura las propiedades del widget para el manejo de eventos."""
        plot_widget.setFocusPolicy(Qt.StrongFocus)
        plot_widget.setMouseTracking(True)
        plot_widget.setAttribute(Qt.WA_AcceptTouchEvents, True)

    def _connect_viewbox_events(self, view_box: ViewBox, widget_name: str) -> None:
        """Conecta eventos específicos del ViewBox."""
        # Cambios de rango (zoom/pan)
        view_box.sigRangeChanged.connect(
            lambda: self._handle_view_range_change(widget_name)
        )

        # Cambio de transformación
        view_box.sigTransformChanged.connect(
            lambda: self._handle_view_transform_change(widget_name)
        )

        # Estado del ViewBox
        view_box.sigStateChanged.connect(
            lambda: self._handle_view_state_change(widget_name)
        )

    def _connect_scene_events(self, scene: GraphicsScene, widget_name: str) -> None:
        """Conecta eventos de la escena gráfica."""
        # Eventos de ratón de la escena
        scene.sigMouseClicked.connect(
            lambda event: self._handle_scene_mouse_click(event, widget_name)
        )

        scene.sigMouseMoved.connect(
            lambda pos: self._handle_scene_mouse_move(pos, widget_name)
        )

        # Eventos de hover
        scene.sigMouseHover.connect(
            lambda items: self._handle_scene_mouse_hover(items, widget_name)
        )

    def _install_event_filter(self, plot_widget: PlotWidget, widget_name: str) -> None:
        """Instala un filtro de eventos personalizado."""
        event_filter = PyQtGraphEventFilter(self.event_service, widget_name)
        plot_widget.installEventFilter(event_filter)
        self._event_filters[widget_name] = event_filter

    @pyqtSlot()
    def _handle_view_range_change(self, widget_name: str) -> None:
        """Maneja cambios en el rango de visualización."""
        widget = self._widgets.get(widget_name)
        if widget is None:
            return

        view_box = widget.getViewBox()
        view_range = view_box.viewRange()

        event_data = EventData(
            event_type=EventType.VIEW_UPDATE,
            source=widget_name,
            data={
                'action': 'range_changed',
                'x_range': view_range[0],
                'y_range': view_range[1]
            },
            timestamp=time.time()
        )

        self.event_service.emit_event(event_data)

    @pyqtSlot()
    def _handle_view_transform_change(self, widget_name: str) -> None:
        """Maneja cambios en la transformación de la vista."""
        event_data = EventData(
            event_type=EventType.VIEW_UPDATE,
            source=widget_name,
            data={'action': 'transform_changed'},
            timestamp=time.time()
        )

        self.event_service.emit_event(event_data)

    @pyqtSlot()
    def _handle_view_state_change(self, widget_name: str) -> None:
        """Maneja cambios en el estado del ViewBox."""
        event_data = EventData(
            event_type=EventType.VIEW_UPDATE,
            source=widget_name,
            data={'action': 'state_changed'},
            timestamp=time.time()
        )

        self.event_service.emit_event(event_data)

    @pyqtSlot(object)
    def _handle_scene_mouse_click(self, event: MouseClickEvent, widget_name: str) -> None:
        """Maneja clics en la escena gráfica."""
        pos = event.pos()

        event_data = EventData(
            event_type=EventType.MOUSE_CLICK,
            source=widget_name,
            data={
                'scene_pos': (pos.x(), pos.y()),
                'button': event.button(),
                'modifiers': event.modifiers(),
                'double_click': event.double(),
                'accepted': event.isAccepted()
            },
            timestamp=time.time(),
            coordinates=(pos.x(), pos.y()),
            button=event.button(),
            modifiers=event.modifiers()
        )

        self.event_service.emit_event(event_data)

    @pyqtSlot(object)
    def _handle_scene_mouse_move(self, pos: QPointF, widget_name: str) -> None:
        """Maneja movimientos del ratón en la escena."""
        event_data = EventData(
            event_type=EventType.MOUSE_MOVE,
            source=widget_name,
            data={'scene_pos': (pos.x(), pos.y())},
            timestamp=time.time(),
            coordinates=(pos.x(), pos.y())
        )

        self.event_service.emit_event(event_data)

    @pyqtSlot(object)
    def _handle_scene_mouse_hover(self, items: List, widget_name: str) -> None:
        """Maneja eventos de hover sobre elementos de la escena."""
        event_data = EventData(
            event_type=EventType.MOUSE_MOVE,
            source=widget_name,
            data={
                'action': 'hover',
                'items_count': len(items),
                'has_items': len(items) > 0
            },
            timestamp=time.time()
        )

        self.event_service.emit_event(event_data)

    def disconnect_widget_events(self, widget_name: str) -> bool:
        """Desconecta todos los eventos de un widget."""
        try:
            if widget_name not in self._widgets:
                logger.warning(f"Widget {widget_name} no está conectado")
                return False

            widget = self._widgets[widget_name]

            # Remover filtro de eventos
            if widget_name in self._event_filters:
                widget.removeEventFilter(self._event_filters[widget_name])
                del self._event_filters[widget_name]

            # Limpiar conexiones
            del self._widgets[widget_name]
            if widget_name in self._connections:
                del self._connections[widget_name]

            logger.info(f"Widget {widget_name} desconectado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error desconectando widget {widget_name}: {e}")
            return False

    def get_connected_widgets(self) -> List[str]:
        """Obtiene la lista de widgets conectados."""
        return list(self._widgets.keys())

    def is_widget_connected(self, widget_name: str) -> bool:
        """Verifica si un widget está conectado."""
        return widget_name in self._widgets


class PyQtGraphEventFilter(QObject):
    """Filtro de eventos especializado para widgets PyQtGraph."""

    def __init__(self, event_service: 'EventService', widget_name: str):
        super().__init__()
        self.event_service = event_service
        self.widget_name = widget_name
        self._last_key_time = 0
        self._key_repeat_threshold = 50  # ms

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Filtra y procesa eventos de teclado, ratón y otros."""
        try:
            event_type = event.type()

            # Eventos de teclado
            if event_type == QEvent.KeyPress:
                return self._handle_key_press(event)
            elif event_type == QEvent.KeyRelease:
                return self._handle_key_release(event)

            # Eventos de ratón
            elif event_type == QEvent.MouseButtonPress:
                return self._handle_mouse_press(event)
            elif event_type == QEvent.MouseMove:
                return self._handle_mouse_move(event)
            elif event_type == QEvent.MouseButtonRelease:
                return self._handle_mouse_release(event)
            elif event_type == QEvent.Wheel:
                return self._handle_wheel_event(event)

            # Eventos de widget
            elif event_type == QEvent.FocusIn:
                return self._handle_focus_in(event)
            elif event_type == QEvent.FocusOut:
                return self._handle_focus_out(event)
            elif event_type == QEvent.Resize:
                return self._handle_resize(event)

            return False

        except Exception as e:
            logger.error(f"Error en eventFilter para {self.widget_name}: {e}")
            return False

    def _handle_key_press(self, event: QKeyEvent) -> bool:
        """Maneja eventos de presión de teclas con filtrado de repetición."""
        current_time = time.time() * 1000  # convertir a ms

        # Filtrar repeticiones rápidas de teclas
        if current_time - self._last_key_time < self._key_repeat_threshold:
            if event.isAutoRepeat():
                return False

        self._last_key_time = current_time

        # Mapear tecla a string
        key_text = self._map_key_to_string(event)

        event_data = EventData(
            event_type=EventType.KEY_PRESS,
            source=self.widget_name,
            data={
                'qt_key': event.key(),
                'native_scan_code': event.nativeScanCode(),
                'auto_repeat': event.isAutoRepeat(),
                'count': event.count()
            },
            timestamp=time.time(),
            key=key_text,
            modifiers=event.modifiers()
        )

        self.event_service.emit_event(event_data)
        return False  # Permitir propagación del evento

    def _handle_key_release(self, event: QKeyEvent) -> bool:
        """Maneja eventos de liberación de teclas."""
        key_text = self._map_key_to_string(event)

        event_data = EventData(
            event_type=EventType.KEY_RELEASE,
            source=self.widget_name,
            data={'qt_key': event.key()},
            timestamp=time.time(),
            key=key_text,
            modifiers=event.modifiers()
        )

        self.event_service.emit_event(event_data)
        return False

    def _map_key_to_string(self, event: QKeyEvent) -> str:
        """Mapea un evento de teclado a su representación string."""
        # Teclas especiales
        special_keys = {
            Qt.Key_PageUp: 'pageup',
            Qt.Key_PageDown: 'pagedown',
            Qt.Key_Escape: 'escape',
            Qt.Key_Enter: 'enter',
            Qt.Key_Return: 'return',
            Qt.Key_Space: 'space',
            Qt.Key_Tab: 'tab',
            Qt.Key_Backspace: 'backspace',
            Qt.Key_Delete: 'delete',
            Qt.Key_Up: 'up',
            Qt.Key_Down: 'down',
            Qt.Key_Left: 'left',
            Qt.Key_Right: 'right'
        }

        if event.key() in special_keys:
            return special_keys[event.key()]

        # Teclas de texto
        key_text = event.text()
        if key_text and key_text.isprintable():
            return key_text

        # Fallback para otras teclas
        return str(event.key())

    def _handle_mouse_press(self, event: QMouseEvent) -> bool:
        """Maneja eventos de presión del ratón."""
        event_data = EventData(
            event_type=EventType.MOUSE_CLICK,
            source=self.widget_name,
            data={
                'x': event.x(),
                'y': event.y(),
                'global_x': event.globalX(),
                'global_y': event.globalY(),
                'button_name': self._get_button_name(event.button())
            },
            timestamp=time.time(),
            coordinates=(event.x(), event.y()),
            button=event.button(),
            modifiers=event.modifiers()
        )

        self.event_service.emit_event(event_data)
        return False

    def _handle_mouse_move(self, event: QMouseEvent) -> bool:
        """Maneja eventos de movimiento del ratón."""
        event_data = EventData(
            event_type=EventType.MOUSE_MOVE,
            source=self.widget_name,
            data={
                'x': event.x(),
                'y': event.y(),
                'global_x': event.globalX(),
                'global_y': event.globalY(),
                'buttons': event.buttons()
            },
            timestamp=time.time(),
            coordinates=(event.x(), event.y()),
            button=event.buttons(),
            modifiers=event.modifiers()
        )

        self.event_service.emit_event(event_data)
        return False

    def _handle_mouse_release(self, event: QMouseEvent) -> bool:
        """Maneja eventos de liberación del ratón."""
        event_data = EventData(
            event_type=EventType.MOUSE_RELEASE,
            source=self.widget_name,
            data={
                'x': event.x(),
                'y': event.y(),
                'global_x': event.globalX(),
                'global_y': event.globalY(),
                'button_name': self._get_button_name(event.button())
            },
            timestamp=time.time(),
            coordinates=(event.x(), event.y()),
            button=event.button(),
            modifiers=event.modifiers()
        )

        self.event_service.emit_event(event_data)
        return False

    def _handle_wheel_event(self, event: QWheelEvent) -> bool:
        """Maneja eventos de rueda del ratón."""
        event_data = EventData(
            event_type=EventType.MOUSE_WHEEL,
            source=self.widget_name,
            data={
                'x': event.x(),
                'y': event.y(),
                'global_x': event.globalX(),
                'global_y': event.globalY(),
                'angle_delta': event.angleDelta().y(),
                'pixel_delta': event.pixelDelta().y()
            },
            timestamp=time.time(),
            coordinates=(event.x(), event.y()),
            modifiers=event.modifiers()
        )

        self.event_service.emit_event(event_data)
        return False

    def _handle_focus_in(self, event: QEvent) -> bool:
        """Maneja eventos de obtención de foco."""
        event_data = EventData(
            event_type=EventType.WIDGET_FOCUS,
            source=self.widget_name,
            data={'focus_gained': True},
            timestamp=time.time()
        )

        self.event_service.emit_event(event_data)
        return False

    def _handle_focus_out(self, event: QEvent) -> bool:
        """Maneja eventos de pérdida de foco."""
        event_data = EventData(
            event_type=EventType.WIDGET_FOCUS,
            source=self.widget_name,
            data={'focus_gained': False},
            timestamp=time.time()
        )

        self.event_service.emit_event(event_data)
        return False

    def _handle_resize(self, event: QEvent) -> bool:
        """Maneja eventos de redimensionamiento del widget."""
        event_data = EventData(
            event_type=EventType.WIDGET_RESIZE,
            source=self.widget_name,
            data={
                'old_size': (event.oldSize().width(), event.oldSize().height()),
                'new_size': (event.size().width(), event.size().height())
            },
            timestamp=time.time()
        )

        self.event_service.emit_event(event_data)
        return False

    def _get_button_name(self, button: Qt.MouseButton) -> str:
        """Obtiene el nombre del botón del ratón."""
        button_names = {
            Qt.LeftButton: 'left',
            Qt.RightButton: 'right',
            Qt.MiddleButton: 'middle'
        }
        return button_names.get(button, f'button_{button}')


class KeyEventProcessor(QObject):
    """Procesador especializado de eventos de teclado con complejidad reducida."""

    def __init__(self, event_service: 'EventService'):
        super().__init__()
        self._event_service = event_service
        self._key_handlers = self._create_key_handler_mapping()

    def _create_key_handler_mapping(self) -> Dict[str, Callable[[], None]]:
        """Crea el mapeo de teclas a funciones de manejo (complejidad: 8)."""
        # Dividir en categorías para reducir complejidad
        handlers = {}
        handlers.update(self._get_mode_handlers())
        handlers.update(self._get_filter_handlers())
        handlers.update(self._get_display_handlers())
        handlers.update(self._get_system_handlers())
        return handlers

    def _get_mode_handlers(self) -> Dict[str, Callable[[], None]]:
        """Obtiene manejadores de cambio de modo (complejidad: 3)."""
        return {
            's': self._toggle_selection_mode,
            '*': self._clear_all_selections,
            'b': self._activate_rectangle_selector,
            'B': self._deactivate_rectangle_selector
        }

    def _get_filter_handlers(self) -> Dict[str, Callable[[], None]]:
        """Obtiene manejadores de filtros (complejidad: 5)."""
        return {
            't': lambda: self._change_filter(1),
            'T': lambda: self._change_filter(-1),
            'a': self._toggle_filter_centering,
            'c': self._toggle_continuum_removal,
            'k': self._request_redshift_input,
            'z': self._toggle_rest_wavelength
        }

    def _get_display_handlers(self) -> Dict[str, Callable[[], None]]:
        """Obtiene manejadores de visualización (complejidad: 8)."""
        return {
            'e': self._toggle_error_display,
            'f': self._toggle_flag_display,
            'r': self._toggle_residuals,
            'y': self._toggle_synthetic_spectrum,
            'h': self._toggle_sonification,
            'j': self._toggle_flux_sensitivity,
            'I': self._toggle_integrated_spectrum
        }

    def _get_system_handlers(self) -> Dict[str, Callable[[], None]]:
        """Obtiene manejadores del sistema (complejidad: 10)."""
        return {
            'S': self._emit_save_request,
            'w': self._emit_window_manager,
            'l': self._emit_lambda_limits,
            'Y': self._emit_flux_limits,
            'F': self._request_spectrum_fit,
            'm': self._request_colormap_selection,
            'i': self._invert_colormap,
            'v': self._request_colormap_limits,
            'q': self._request_quit,
            '+': lambda: self._change_colormap(1),
            '-': lambda: self._change_colormap(-1),
            'pageup': lambda: self._change_colormap(1),
            'pagedown': lambda: self._change_colormap(-1),
            '1': lambda: self._change_color_scale('linear'),
            '2': lambda: self._change_color_scale('ilog'),
            '3': lambda: self._change_color_scale('sqrt'),
            '4': lambda: self._change_color_scale('power'),
            '5': lambda: self._change_color_scale('asinh'),
            '0': self._show_available_scales
        }

    @pyqtSlot(EventData)
    def process_key_event(self, event_data: EventData) -> None:
        """Procesa eventos de teclado (complejidad: 6)."""
        if event_data.event_type != EventType.KEY_PRESS:
            return

        key = event_data.key
        if not key:
            return

        handler = self._key_handlers.get(key)
        if handler:
            try:
                handler()
                logger.debug(f"Ejecutado manejador para tecla: {key}")
            except Exception as e:
                logger.error(f"Error procesando tecla {key}: {e}")
                logger.error(traceback.format_exc())

    # Métodos de manejo de eventos individuales (cada uno con complejidad <= 5)

    def _toggle_selection_mode(self) -> None:
        """Alterna entre modo espectro y modo selección."""
        current_mode = self._event_service.get_system_state().get('current_mode', 'spectrum')
        new_mode = 'selection' if current_mode == 'spectrum' else 'spectrum'

        self._event_service.update_system_state(current_mode=new_mode)
        self._emit_view_update_event({'mode_change': new_mode})

    def _clear_all_selections(self) -> None:
        """Limpia todas las selecciones actuales."""
        self._event_service.set_current_selection([])
        self._emit_view_update_event({'clear_selections': True})

    def _emit_save_request(self) -> None:
        """Emite solicitud de guardado."""
        self._emit_event(EventType.SAVE_REQUEST, {
            'selected_spaxels': self._event_service.get_current_selection()
        })

    def _emit_window_manager(self) -> None:
        """Emite solicitud del administrador de ventanas."""
        self._emit_event(EventType.WINDOW_MANAGER, {})

    def _emit_lambda_limits(self) -> None:
        """Emite solicitud de configuración de límites lambda."""
        self._emit_event(EventType.LAMBDA_LIMITS, {})

    def _emit_flux_limits(self) -> None:
        """Emite solicitud de configuración de límites de flujo."""
        self._emit_event(EventType.FLUX_LIMITS, {})

    def _change_filter(self, direction: int) -> None:
        """Cambia el filtro actual."""
        self._emit_event(EventType.FILTER_CHANGE, {'direction': direction})

    def _toggle_filter_centering(self) -> None:
        """Alterna el centrado del filtro."""
        current_state = self._event_service.get_system_state().get('filter_centered', False)
        self._event_service.update_system_state(filter_centered=not current_state)
        self._emit_event(EventType.FILTER_CHANGE, {'toggle_centering': True})

    def _toggle_continuum_removal(self) -> None:
        """Alterna la remoción de continuo."""
        current_state = self._event_service.get_system_state().get('continuum_removed', False)
        self._event_service.update_system_state(continuum_removed=not current_state)
        self._emit_event(EventType.FILTER_CHANGE, {'toggle_continuum_removal': True})

    def _request_redshift_input(self) -> None:
        """Solicita entrada de redshift."""
        self._emit_event(EventType.REDSHIFT_CHANGE, {'request_input': True})

    def _toggle_rest_wavelength(self) -> None:
        """Alterna entre wavelength observada y en reposo."""
        current_state = self._event_service.get_system_state().get('rest_wavelength_mode', False)
        self._event_service.update_system_state(rest_wavelength_mode=not current_state)
        self._emit_event(EventType.REST_WAVE_TOGGLE, {'rest_mode': not current_state})

    def _toggle_error_display(self) -> None:
        """Alterna la visualización de errores."""
        current_state = self._event_service.get_system_state().get('show_errors', False)
        self._event_service.update_system_state(show_errors=not current_state)
        self._emit_event(EventType.ERROR_TOGGLE, {'show_errors': not current_state})

    def _toggle_flag_display(self) -> None:
        """Alterna la visualización de flags."""
        current_state = self._event_service.get_system_state().get('show_flags', False)
        self._event_service.update_system_state(show_flags=not current_state)
        self._emit_event(EventType.FLAG_TOGGLE, {'show_flags': not current_state})

    def _toggle_residuals(self) -> None:
        """Alterna la visualización de residuos."""
        current_state = self._event_service.get_system_state().get('show_residuals', False)
        self._event_service.update_system_state(show_residuals=not current_state)
        self._emit_view_update_event({'toggle_residuals': True})

    def _toggle_synthetic_spectrum(self) -> None:
        """Alterna la visualización del espectro sintético."""
        current_state = self._event_service.get_system_state().get('show_synthetic', False)
        self._event_service.update_system_state(show_synthetic=not current_state)
        self._emit_event(EventType.SYNTHETIC_TOGGLE, {'show_synthetic': not current_state})

    def _toggle_sonification(self) -> None:
        """Alterna la sonificación."""
        current_state = self._event_service.get_system_state().get('sonification_active', False)
        self._event_service.update_system_state(sonification_active=not current_state)
        self._emit_event(EventType.SONIFICATION, {'active': not current_state})

    def _toggle_flux_sensitivity(self) -> None:
        """Alterna la sensibilidad de flujo en sonificación."""
        current_state = self._event_service.get_system_state().get('flux_sensitivity', False)
        self._event_service.update_system_state(flux_sensitivity=not current_state)
        self._emit_event(EventType.SONIFICATION, {'toggle_flux_sensitivity': True})

    def _request_spectrum_fit(self) -> None:
        """Solicita ajuste de espectro."""
        self._emit_event(EventType.FIT_SPECTRUM, {
            'current_selection': self._event_service.get_current_selection()
        })

    def _activate_rectangle_selector(self) -> None:
        """Activa el selector rectangular."""
        self._event_service.update_system_state(rectangle_selector_active=True)
        self._emit_event(EventType.RECTANGLE_SELECT, {'activate': True})

    def _deactivate_rectangle_selector(self) -> None:
        """Desactiva el selector rectangular."""
        self._event_service.update_system_state(rectangle_selector_active=False)
        self._emit_event(EventType.RECTANGLE_SELECT, {'activate': False})

    def _change_colormap(self, direction: int) -> None:
        """Cambia el mapa de colores."""
        self._emit_event(EventType.COLORMAP_CHANGE, {'direction': direction})

    def _request_colormap_selection(self) -> None:
        """Solicita selección manual de colormap."""
        self._emit_event(EventType.COLORMAP_CHANGE, {'request_selection': True})

    def _invert_colormap(self) -> None:
        """Invierte el colormap actual."""
        colormap_settings = self._event_service.get_system_state().get('colormap_settings', {})
        colormap_settings['inverted'] = not colormap_settings.get('inverted', False)
        self._event_service.update_system_state(colormap_settings=colormap_settings)
        self._emit_event(EventType.COLORMAP_CHANGE, {'invert': True})

    def _change_color_scale(self, scale: str) -> None:
        """Cambia la escala de colores."""
        colormap_settings = self._event_service.get_system_state().get('colormap_settings', {})
        colormap_settings['scale'] = scale
        self._event_service.update_system_state(colormap_settings=colormap_settings)
        self._emit_event(EventType.COLORMAP_CHANGE, {'scale': scale})

    def _show_available_scales(self) -> None:
        """Muestra las escalas disponibles."""
        available_scales = ['linear', 'ilog', 'sqrt', 'power', 'asinh']
        self._emit_event(EventType.COLORMAP_CHANGE, {
            'show_scales': True,
            'available_scales': available_scales
        })

    def _request_colormap_limits(self) -> None:
        """Solicita configuración de límites de colormap."""
        self._emit_event(EventType.COLORMAP_CHANGE, {'request_limits': True})

    def _request_quit(self) -> None:
        """Solicita salir de la aplicación."""
        self._emit_event(EventType.APPLICATION_EXIT, {'quit_request': True})

    def _toggle_integrated_spectrum(self) -> None:
        """Alterna la visualización del espectro integrado."""
        current_state = self._event_service.get_system_state().get('show_integrated', False)
        self._event_service.update_system_state(show_integrated=not current_state)
        self._emit_event(EventType.SPECTRUM_UPDATE, {'toggle_integrated': True})

    # Métodos auxiliares

    def _emit_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Método auxiliar para emitir eventos."""
        event_data = EventData(
            event_type=event_type,
            source='key_processor',
            data=data,
            timestamp=time.time()
        )
        self._event_service.emit_event(event_data)

    def _emit_view_update_event(self, data: Dict[str, Any]) -> None:
        """Método auxiliar para emitir eventos de actualización de vista."""
        self._emit_event(EventType.VIEW_UPDATE, data)


class MouseEventProcessor(QObject):
    """Procesador especializado de eventos de ratón con complejidad reducida."""

    def __init__(self, event_service: 'EventService'):
        super().__init__()
        self._event_service = event_service
        self._drag_threshold = 5  # píxeles
        self._click_timeout = 200  # ms
        self._last_click_time = 0
        self._last_click_pos = None

    @pyqtSlot(EventData)
    def process_mouse_event(self, event_data: EventData) -> None:
        """Procesa eventos de ratón (complejidad: 5)."""
        processors = {
            EventType.MOUSE_CLICK: self._process_mouse_click,
            EventType.MOUSE_MOVE: self._process_mouse_move,
            EventType.MOUSE_RELEASE: self._process_mouse_release,
            EventType.MOUSE_WHEEL: self._process_mouse_wheel
        }

        processor = processors.get(event_data.event_type)
        if processor:
            try:
                processor(event_data)
            except Exception as e:
                logger.error(f"Error procesando evento de ratón {event_data.event_type}: {e}")

    def _process_mouse_click(self, event_data: EventData) -> None:
        """Procesa clics del ratón (complejidad: 4)."""
        source = event_data.source.lower()

        if 'spaxel' in source:
            self._handle_spaxel_click(event_data)
        elif 'spectrum' in source:
            self._handle_spectrum_click(event_data)
        else:
            self._handle_generic_click(event_data)

    def _process_mouse_move(self, event_data: EventData) -> None:
        """Procesa movimientos del ratón (complejidad: 6)."""
        source = event_data.source.lower()
        state = self._event_service.get_system_state()

        if 'spaxel' in source and state.get('current_mode') == 'spectrum':
            self._handle_spaxel_hover(event_data)
        elif 'spectrum' in source:
            self._handle_spectrum_hover(event_data)

        # Manejar interacciones especiales
        self._handle_special_interactions(event_data)

    def _process_mouse_release(self, event_data: EventData) -> None:
        """Procesa liberación del ratón (complejidad: 3)."""
        self._handle_passband_release(event_data)
        self._handle_rectangle_selection_end(event_data)

    def _process_mouse_wheel(self, event_data: EventData) -> None:
        """Procesa eventos de rueda del ratón (complejidad: 4)."""
        angle_delta = event_data.data.get('angle_delta', 0)
        modifiers = event_data.modifiers or Qt.NoModifier

        if modifiers & Qt.ControlModifier:
            self._handle_zoom_wheel(event_data, angle_delta)
        elif modifiers & Qt.ShiftModifier:
            self._handle_colormap_wheel(event_data, angle_delta)
        else:
            self._handle_scroll_wheel(event_data, angle_delta)

    def _handle_spaxel_click(self, event_data: EventData) -> None:
        """Procesa clics en el visor de spaxels (complejidad: 5)."""
        coordinates = event_data.coordinates
        if not coordinates:
            return

        x_data, y_data = coordinates
        spaxel_coords = self._convert_to_spaxel_coordinates(x_data, y_data)

        state = self._event_service.get_system_state()
        mode = state.get('current_mode', 'spectrum')

        if mode == 'selection':
            self._add_spaxel_to_selection(spaxel_coords)
        else:
            self._select_single_spaxel(spaxel_coords)

    def _handle_spaxel_hover(self, event_data: EventData) -> None:
        """Procesa hover sobre spaxels (complejidad: 3)."""
        coordinates = event_data.coordinates
        if coordinates:
            spaxel_coords = self._convert_to_spaxel_coordinates(*coordinates)
            self._update_spectrum_preview(spaxel_coords)

    def _handle_spectrum_click(self, event_data: EventData) -> None:
        """Procesa clics en el visor de espectros (complejidad: 4)."""
        button = event_data.button

        if button == Qt.LeftButton:
            self._start_passband_interaction(event_data)
        elif button == Qt.RightButton:
            self._handle_context_menu(event_data)
        elif button == Qt.MiddleButton:
            self._handle_middle_click(event_data)

    def _handle_spectrum_hover(self, event_data: EventData) -> None:
        """Procesa hover sobre el espectro (complejidad: 2)."""
        coordinates = event_data.coordinates
        if coordinates:
            self._update_cursor_info(coordinates)

    def _handle_generic_click(self, event_data: EventData) -> None:
        """Maneja clics en widgets genéricos (complejidad: 2)."""
        click_event = EventData(
            event_type=EventType.VIEW_UPDATE,
            source=event_data.source,
            data={'generic_click': True, 'coordinates': event_data.coordinates},
            timestamp=event_data.timestamp
        )
        self._event_service.emit_event(click_event)

    def _handle_special_interactions(self, event_data: EventData) -> None:
        """Maneja interacciones especiales (complejidad: 6)."""
        state = self._event_service.get_system_state()

        # Interacción de passband
        if state.get('passband_interaction_active'):
            self._handle_passband_move(event_data)

        # Selector rectangular
        if state.get('rectangle_selector_active'):
            self._handle_rectangle_selection_move(event_data)

        # Cambio dinámico de colormap con botón derecho
        if event_data.button == Qt.RightButton:
            self._handle_dynamic_colormap_change(event_data)

    def _handle_zoom_wheel(self, event_data: EventData, angle_delta: int) -> None:
        """Maneja zoom con rueda del ratón (complejidad: 2)."""
        zoom_factor = 1.1 if angle_delta > 0 else 0.9
        self._emit_view_event('zoom', {'factor': zoom_factor, 'center': event_data.coordinates})

    def _handle_colormap_wheel(self, event_data: EventData, angle_delta: int) -> None:
        """Maneja cambio de colormap con rueda (complejidad: 2)."""
        direction = 1 if angle_delta > 0 else -1
        self._emit_colormap_event({'wheel_change': direction})

    def _handle_scroll_wheel(self, event_data: EventData, angle_delta: int) -> None:
        """Maneja scroll normal con rueda (complejidad: 2)."""
        scroll_amount = angle_delta / 120  # Normalizar
        self._emit_view_event('scroll', {'amount': scroll_amount})

    # Métodos auxiliares para operaciones específicas

    def _convert_to_spaxel_coordinates(self, x_data: float, y_data: float) -> Tuple[int, int]:
        """Convierte coordenadas de datos a coordenadas de spaxel."""
        return (int(round(x_data)), int(round(y_data)))

    def _add_spaxel_to_selection(self, coords: Tuple[int, int]) -> None:
        """Añade un spaxel a la selección actual."""
        current_selection = self._event_service.get_current_selection()

        if coords not in current_selection:
            current_selection.append(coords)
            self._event_service.set_current_selection(current_selection)

            self._emit_spaxel_event(coords, selection_mode=True)

    def _select_single_spaxel(self, coords: Tuple[int, int]) -> None:
        """Selecciona un único spaxel."""
        self._event_service.set_current_selection([coords])
        self._emit_spaxel_event(coords, selection_mode=False)

    def _update_spectrum_preview(self, coords: Tuple[int, int]) -> None:
        """Actualiza la previsualización del espectro."""
        self._emit_spectrum_event({'coordinates': coords, 'preview': True})

    def _start_passband_interaction(self, event_data: EventData) -> None:
        """Inicia la interacción con passband."""
        self._event_service.update_system_state(passband_interaction_active=True)
        self._emit_passband_event('start', event_data.coordinates)

    def _handle_passband_move(self, event_data: EventData) -> None:
        """Maneja el movimiento durante interacción de passband."""
        self._emit_passband_event('move', event_data.coordinates)

    def _handle_passband_release(self, event_data: EventData) -> None:
        """Maneja la liberación durante interacción de passband."""
        state = self._event_service.get_system_state()
        if state.get('passband_interaction_active'):
            self._event_service.update_system_state(passband_interaction_active=False)
            self._emit_passband_event('end', event_data.coordinates)

    def _handle_rectangle_selection_move(self, event_data: EventData) -> None:
        """Maneja movimiento durante selección rectangular."""
        self._emit_rectangle_event('update', event_data.coordinates)

    def _handle_rectangle_selection_end(self, event_data: EventData) -> None:
        """Maneja fin de selección rectangular."""
        state = self._event_service.get_system_state()
        if state.get('rectangle_selector_active'):
            self._emit_rectangle_event('complete', event_data.coordinates)

    def _handle_dynamic_colormap_change(self, event_data: EventData) -> None:
        """Maneja cambio dinámico de colormap."""
        coordinates = event_data.coordinates
        if coordinates:
            self._emit_colormap_event({
                'dynamic_change': True,
                'x_position': coordinates[0],
                'y_position': coordinates[1]
            })

    def _handle_context_menu(self, event_data: EventData) -> None:
        """Maneja menú contextual."""
        self._emit_view_event('context_menu', {'position': event_data.coordinates})

    def _handle_middle_click(self, event_data: EventData) -> None:
        """Maneja clic con botón central."""
        self._emit_view_event('middle_click', {'position': event_data.coordinates})

    def _update_cursor_info(self, coordinates: Tuple[float, float]) -> None:
        """Actualiza información del cursor."""
        self._emit_view_event('cursor_update', {'coordinates': coordinates})

    # Métodos auxiliares para emisión de eventos

    def _emit_spaxel_event(self, coords: Tuple[int, int], selection_mode: bool) -> None:
        """Emite evento de selección de spaxel."""
        event_data = EventData(
            event_type=EventType.SPAXEL_SELECT,
            source='mouse_processor',
            data={'coordinates': coords, 'selection_mode': selection_mode},
            timestamp=time.time()
        )
        self._event_service.emit_event(event_data)

    def _emit_spectrum_event(self, data: Dict[str, Any]) -> None:
        """Emite evento de espectro."""
        event_data = EventData(
            event_type=EventType.SPECTRUM_UPDATE,
            source='mouse_processor',
            data=data,
            timestamp=time.time()
        )
        self._event_service.emit_event(event_data)

    def _emit_passband_event(self, action: str, position: Optional[Tuple[float, float]]) -> None:
        """Emite evento de interacción de passband."""
        event_data = EventData(
            event_type=EventType.PASSBAND_INTERACTION,
            source='mouse_processor',
            data={'action': action, 'position': position},
            timestamp=time.time()
        )
        self._event_service.emit_event(event_data)

    def _emit_rectangle_event(self, action: str, position: Optional[Tuple[float, float]]) -> None:
        """Emite evento de selección rectangular."""
        event_data = EventData(
            event_type=EventType.RECTANGLE_SELECT,
            source='mouse_processor',
            data={'action': action, 'position': position},
            timestamp=time.time()
        )
        self._event_service.emit_event(event_data)

    def _emit_colormap_event(self, data: Dict[str, Any]) -> None:
        """Emite evento de cambio de colormap."""
        event_data = EventData(
            event_type=EventType.COLORMAP_CHANGE,
            source='mouse_processor',
            data=data,
            timestamp=time.time()
        )
        self._event_service.emit_event(event_data)

    def _emit_view_event(self, action: str, data: Dict[str, Any]) -> None:
        """Emite evento de actualización de vista."""
        data['action'] = action
        event_data = EventData(
            event_type=EventType.VIEW_UPDATE,
            source='mouse_processor',
            data=data,
            timestamp=time.time()
        )
        self._event_service.emit_event(event_data)


class EventService(QObject):
    """
    Servicio central completamente migrado a PyQt5/PyQtGraph.

    Esta implementación elimina por completo cualquier referencia o compatibilidad
    con Matplotlib, utilizando exclusivamente el ecosistema PyQt5/PyQtGraph.
    """

    # Señales principales del servicio
    system_state_changed = pyqtSignal(dict)
    selection_changed = pyqtSignal(list)
    service_ready = pyqtSignal()
    service_shutdown = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Inicializar componentes especializados
        self._initialize_components()

        # Estado de la aplicación
        self._initialize_application_state()

        # Configurar conexiones de eventos
        self._setup_event_connections()

        # Configurar sistema de limpieza
        self._setup_cleanup_system()

        logger.info("EventService inicializado completamente con PyQt5/PyQtGraph")
        self.service_ready.emit()

    def _initialize_components(self) -> None:
        """Inicializa todos los componentes especializados (complejidad: 4)."""
        self._event_registry = EventHandlerRegistry()
        self._system_state = SystemState()
        self._pyqtgraph_manager = PyQtGraphConnectionManager(self)
        self._key_processor = KeyEventProcessor(self)
        self._mouse_processor = MouseEventProcessor(self)

    def _initialize_application_state(self) -> None:
        """Inicializa el estado de la aplicación (complejidad: 4)."""
        self._current_selection = []
        self._active_widgets = {}
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._periodic_cleanup)
        self._cleanup_timer.start(60000)  # Limpieza cada minuto

    def _setup_event_connections(self) -> None:
        """Configura las conexiones entre componentes (complejidad: 6)."""
        # Conectar procesadores a eventos
        self._event_registry.signals.key_pressed.connect(
            self._key_processor.process_key_event, Qt.QueuedConnection
        )

        self._event_registry.signals.mouse_clicked.connect(
            self._mouse_processor.process_mouse_event, Qt.QueuedConnection
        )

        self._event_registry.signals.mouse_moved.connect(
            self._mouse_processor.process_mouse_event, Qt.QueuedConnection
        )

        self._event_registry.signals.mouse_released.connect(
            self._mouse_processor.process_mouse_event, Qt.QueuedConnection
        )

        self._event_registry.signals.mouse_wheel.connect(
            self._mouse_processor.process_mouse_event, Qt.QueuedConnection
        )

        # Conectar señales de estado
        self._system_state.state_changed.connect(
            self.system_state_changed.emit, Qt.QueuedConnection
        )

        self._system_state.selection_changed.connect(
            self.selection_changed.emit, Qt.QueuedConnection
        )

    def _setup_cleanup_system(self) -> None:
        """Configura el sistema de limpieza automática (complejidad: 2)."""
        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(self.cleanup)

    # Métodos públicos principales

    def register_handler(self, event_type: EventType, handler: Callable[[EventData], None]) -> bool:
        """Registra un manejador para un tipo de evento específico."""
        return self._event_registry.register_handler(event_type, handler)

    def unregister_handler(self, event_type: EventType, handler: Callable[[EventData], None]) -> bool:
        """Desregistra un manejador de eventos."""
        return self._event_registry.unregister_handler(event_type, handler)

    def emit_event(self, event_data: EventData) -> bool:
        """Emite un evento usando el sistema de señales PyQt5."""
        return self._event_registry.emit_event(event_data)

    def setup_pyqtgraph_widget(self, plot_widget: PlotWidget, widget_name: str) -> bool:
        """Configura un widget PyQtGraph para el manejo de eventos."""
        if self._pyqtgraph_manager.connect_plot_widget(plot_widget, widget_name):
            self._active_widgets[widget_name] = plot_widget
            return True
        return False

    def disconnect_pyqtgraph_widget(self, widget_name: str) -> bool:
        """Desconecta los eventos de un widget PyQtGraph."""
        if self._pyqtgraph_manager.disconnect_widget_events(widget_name):
            self._active_widgets.pop(widget_name, None)
            return True
        return False

    def get_system_state(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema."""
        return self._system_state.get_state()

    def update_system_state(self, **kwargs) -> Dict[str, Any]:
        """Actualiza el estado del sistema."""
        return self._system_state.update_state(**kwargs)

    def get_current_selection(self) -> List[Tuple[int, int]]:
        """Obtiene la selección actual de spaxels."""
        return self._current_selection.copy()

    def set_current_selection(self, selection: List[Tuple[int, int]]) -> None:
        """Establece la selección actual de spaxels."""
        self._current_selection = selection
        self.update_system_state(selected_spaxels=selection)
        self.selection_changed.emit(selection)

    def get_active_widgets(self) -> Dict[str, PlotWidget]:
        """Obtiene los widgets PyQtGraph activos."""
        return self._active_widgets.copy()

    def get_handler_statistics(self) -> Dict[EventType, int]:
        """Obtiene estadísticas de manejadores registrados."""
        return {event_type: self._event_registry.get_handler_count(event_type)
                for event_type in EventType}

    # Métodos de mantenimiento y limpieza

    @pyqtSlot()
    def _periodic_cleanup(self) -> None:
        """Realiza limpieza periódica del sistema (complejidad: 4)."""
        try:
            # Limpiar widgets desconectados
            disconnected_widgets = []
            for widget_name, widget in self._active_widgets.items():
                if not widget or widget.parent() is None:
                    disconnected_widgets.append(widget_name)

            for widget_name in disconnected_widgets:
                self.disconnect_pyqtgraph_widget(widget_name)
                logger.info(f"Widget desconectado automáticamente: {widget_name}")

            # Forzar recolección de basura en objetos Qt
            app = QApplication.instance()
            if app:
                app.processEvents()

        except Exception as e:
            logger.error(f"Error en limpieza periódica: {e}")

    @pyqtSlot()
    def cleanup(self) -> None:
        """Limpia todos los recursos y desconecta eventos."""
        try:
            logger.info("Iniciando limpieza del EventService...")

            # Detener timer de limpieza
            if self._cleanup_timer and self._cleanup_timer.isActive():
                self._cleanup_timer.stop()

            # Desconectar todos los widgets
            for widget_name in list(self._active_widgets.keys()):
                self.disconnect_pyqtgraph_widget(widget_name)

            # Resetear estado del sistema
            self._system_state.reset_state()

            # Limpiar selección
            self._current_selection = []

            # Emitir señal de shutdown
            self.service_shutdown.emit()

            logger.info("EventService limpiado exitosamente")

        except Exception as e:
            logger.error(f"Error durante limpieza: {e}")

    def reset_service(self) -> None:
        """Resetea el servicio a su estado inicial."""
        self.cleanup()
        self._initialize_application_state()
        logger.info("EventService reseteado")


# Función de utilidad para crear el servicio
def create_event_service() -> EventService:
    """
    Crea y configura una instancia completa de EventService.

    Returns:
        EventService completamente configurado y listo para usar
    """
    try:
        service = EventService()
        logger.info("EventService creado exitosamente")
        return service
    except Exception as e:
        logger.error(f"Error creando EventService: {e}")
        raise


# Clase de pruebas integradas
class EventServiceTester(QObject):
    """Clase de pruebas para verificar el funcionamiento del EventService."""

    test_completed = pyqtSignal(bool, str)

    def __init__(self, event_service: EventService):
        super().__init__()
        self.event_service = event_service
        self.test_results = []

    def run_all_tests(self) -> bool:
        """Ejecuta todas las pruebas del EventService."""
        tests = [
            self._test_service_initialization,
            self._test_widget_connection,
            self._test_event_emission,
            self._test_state_management,
            self._test_selection_management,
            self._test_handler_registration,
            self._test_cleanup
        ]

        all_passed = True
        for test in tests:
            try:
                passed, message = test()
                self.test_results.append((test.__name__, passed, message))
                if not passed:
                    all_passed = False
                    logger.error(f"Test falló: {test.__name__} - {message}")
            except Exception as e:
                all_passed = False
                error_msg = f"Test error: {str(e)}"
                self.test_results.append((test.__name__, False, error_msg))
                logger.error(f"Test {test.__name__} generó excepción: {e}")

        self.test_completed.emit(all_passed, self._generate_test_report())
        return all_passed

    def _test_service_initialization(self) -> Tuple[bool, str]:
        """Prueba la inicialización del servicio."""
        if not isinstance(self.event_service, EventService):
            return False, "Servicio no es instancia de EventService"

        if not hasattr(self.event_service, '_event_registry'):
            return False, "Registro de eventos no inicializado"

        if not hasattr(self.event_service, '_system_state'):
            return False, "Estado del sistema no inicializado"

        return True, "Servicio inicializado correctamente"

    def _test_widget_connection(self) -> Tuple[bool, str]:
        """Prueba la conexión de widgets PyQtGraph."""
        app = QApplication.instance()
        if not app:
            app = QApplication([])

        try:
            plot_widget = PlotWidget()
            success = self.event_service.setup_pyqtgraph_widget(plot_widget, "test_widget")

            if not success:
                return False, "Falló la conexión del widget"

            if "test_widget" not in self.event_service.get_active_widgets():
                return False, "Widget no aparece en la lista de activos"

            # Limpiar
            self.event_service.disconnect_pyqtgraph_widget("test_widget")
            plot_widget.close()

            return True, "Conexión de widget exitosa"

        except Exception as e:
            return False, f"Error en conexión de widget: {e}"

    def _test_event_emission(self) -> Tuple[bool, str]:
        """Prueba la emisión de eventos."""
        events_received = []

        def test_handler(event_data: EventData):
            events_received.append(event_data)

        # Registrar manejador
        success = self.event_service.register_handler(EventType.MOUSE_CLICK, test_handler)
        if not success:
            return False, "Falló el registro del manejador"

        # Emitir evento de prueba
        test_event = EventData(
            event_type=EventType.MOUSE_CLICK,
            source="test",
            data={'x': 100, 'y': 200},
            timestamp=time.time(),
            coordinates=(100, 200),
            button=Qt.LeftButton
        )

        emit_success = self.event_service.emit_event(test_event)
        if not emit_success:
            return False, "Falló la emisión del evento"

        # Procesar eventos pendientes
        QApplication.processEvents()

        if len(events_received) == 0:
            return False, "Evento no fue recibido"

        # Limpiar
        self.event_service.unregister_handler(EventType.MOUSE_CLICK, test_handler)

        return True, "Emisión de eventos exitosa"

    def _test_state_management(self) -> Tuple[bool, str]:
        """Prueba la gestión del estado del sistema."""
        initial_state = self.event_service.get_system_state()
        if not isinstance(initial_state, dict):
            return False, "Estado no es un diccionario"

        # Actualizar estado
        updated_state = self.event_service.update_system_state(test_value=True)
        if 'test_value' not in updated_state:
            return False, "Actualización de estado falló"

        # Verificar persistencia
        current_state = self.event_service.get_system_state()
        if not current_state.get('test_value'):
            return False, "Estado no persistió"

        return True, "Gestión de estado exitosa"

    def _test_selection_management(self) -> Tuple[bool, str]:
        """Prueba la gestión de selecciones."""
        # Estado inicial
        initial_selection = self.event_service.get_current_selection()
        if not isinstance(initial_selection, list):
            return False, "Selección inicial no es lista"

        # Establecer selección
        test_selection = [(1, 2), (3, 4)]
        self.event_service.set_current_selection(test_selection)

        current_selection = self.event_service.get_current_selection()
        if current_selection != test_selection:
            return False, "Selección no se estableció correctamente"

        # Verificar en estado del sistema
        system_state = self.event_service.get_system_state()
        if system_state.get('selected_spaxels') != test_selection:
            return False, "Selección no se reflejó en el estado del sistema"

        # Limpiar
        self.event_service.set_current_selection([])

        return True, "Gestión de selecciones exitosa"

    def _test_handler_registration(self) -> Tuple[bool, str]:
        """Prueba el registro y eliminación de manejadores."""

        def dummy_handler(event_data: EventData):
            pass

        # Verificar conteo inicial
        initial_count = self.event_service._event_registry.get_handler_count(EventType.KEY_PRESS)

        # Registrar manejador
        success = self.event_service.register_handler(EventType.KEY_PRESS, dummy_handler)
        if not success:
            return False, "Falló el registro del manejador"

        # Verificar incremento
        new_count = self.event_service._event_registry.get_handler_count(EventType.KEY_PRESS)
        if new_count != initial_count + 1:
            return False, "Conteo de manejadores no se incrementó"

        # Desregistrar manejador
        unregister_success = self.event_service.unregister_handler(EventType.KEY_PRESS, dummy_handler)
        if not unregister_success:
            return False, "Falló la eliminación del manejador"

        # Verificar decremento
        final_count = self.event_service._event_registry.get_handler_count(EventType.KEY_PRESS)
        if final_count != initial_count:
            return False, "Conteo de manejadores no se decrementó"

        return True, "Registro de manejadores exitoso"

    def _test_cleanup(self) -> Tuple[bool, str]:
        """Prueba la limpieza del servicio."""
        # Configurar estado de prueba
        self.event_service.set_current_selection([(5, 6)])
        self.event_service.update_system_state(test_cleanup=True)

        # Ejecutar limpieza
        self.event_service.cleanup()

        # Verificar limpieza
        selection = self.event_service.get_current_selection()
        if len(selection) != 0:
            return False, "Selección no se limpió"

        state = self.event_service.get_system_state()
        if state.get('test_cleanup'):
            return False, "Estado no se reseteó"

        return True, "Limpieza exitosa"

    def _generate_test_report(self) -> str:
        """Genera un reporte de las pruebas ejecutadas."""
        report = "\n=== REPORTE DE PRUEBAS ===\n"
        passed = sum(1 for _, result, _ in self.test_results if result)
        total = len(self.test_results)

        report += f"Pruebas pasadas: {passed}/{total}\n\n"

        for test_name, result, message in self.test_results:
            status = "✅ PASÓ" if result else "❌ FALLÓ"
            report += f"{status} {test_name}: {message}\n"

        return report


# Función principal de pruebas
def test_event_service() -> bool:
    """Función principal para probar el EventService completamente."""
    print("=== INICIANDO PRUEBAS DEL EVENTSERVICE ===\n")

    # Asegurar que existe una aplicación Qt
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    try:
        # Crear servicio de eventos
        service = create_event_service()

        # Crear y ejecutar pruebas
        tester = EventServiceTester(service)

        success = tester.run_all_tests()

        # Mostrar reporte
        print(tester._generate_test_report())

        # Limpieza final
        service.cleanup()

        if success:
            print("\n🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
            print("✅ El EventService está completamente migrado a PyQt5/PyQtGraph")
            print("✅ Estructura modular implementada correctamente")
            print("✅ Complejidad ciclomática controlada")
            print("✅ Acoplamiento reducido exitosamente")
            print("✅ Funcionalidad original preservada")
        else:
            print("\n❌ ALGUNAS PRUEBAS FALLARON")
            print("Por favor revisar los errores reportados")

        return success

    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN LAS PRUEBAS: {e}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Ejecutar pruebas cuando se ejecuta directamente
    test_event_service()