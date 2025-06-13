from PyQt5.QtCore import QObject, pyqtSignal
from ...core.services import EventService


class EventController(QObject):
    """Controlador centralizado para gestión de eventos"""

    spaxel_selected = pyqtSignal(int, int)
    key_pressed = pyqtSignal(str)

    def __init__(self, event_service: EventService):
        super().__init__()
        self.event_service = event_service
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Configura los manejadores de eventos del servicio"""
        self.event_service.register_handler("mouse_click", self._handle_mouse_click)
        self.event_service.register_handler("key_press", self._handle_key_press)

    def _handle_mouse_click(self, event_data):
        """Procesa eventos de clic de ratón"""
        if event_data['source'] == 'spaxel_viewer':
            self.spaxel_selected.emit(event_data['x'], event_data['y'])

    def _handle_key_press(self, event_data):
        """Procesa eventos de teclado"""
        self.key_pressed.emit(event_data['key'])