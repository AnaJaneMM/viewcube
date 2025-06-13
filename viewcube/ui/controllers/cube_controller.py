from PyQt5.QtCore import QObject, pyqtSignal
from ...core.services import FilterService
from ...adapters.presenters import CubePresenter


class CubeController(QObject):
    """Controlador específico para operaciones de cubo 3D"""

    slice_changed = pyqtSignal(int)

    def __init__(self, data_service, event_service):
        super().__init__()
        self.data_service = data_service
        self.event_service = event_service
        self.filter_service = FilterService()

        self.current_filter = None
        self._connect_events()

    def _connect_events(self):
        """Registra manejadores de eventos"""
        self.event_service.register_handler("filter_change", self.apply_filter)
        self.event_service.register_handler("slice_change", self.update_slice)

    def apply_filter(self, filter_name: str):
        """Aplica un filtro al cubo actual"""
        filter_data = self.filter_service.load_filter(filter_name)
        filtered_cube = self.filter_service.apply_to_cube(
            self.data_service.current_cube,
            filter_data
        )
        self.data_service.update_cube(filtered_cube)
        self.slice_changed.emit(self.data_service.current_slice)

    def update_slice(self, slice_idx: int):
        """Actualiza el slice visible del cubo"""
        self.data_service.current_slice = slice_idx
        self.slice_changed.emit(slice_idx)