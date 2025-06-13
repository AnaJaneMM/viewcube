from PyQt5.QtCore import Qt, QObject
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QSplitter
import pyqtgraph as pg

from ...adapters.repositories import FitsRepository, ConfigRepository
from ...core.services import DataService, EventService
from ...adapters.presenters import CubePresenter, SpectrumPresenter

class MainController(QMainWindow):
    """Controlador principal que coordina todas las vistas y servicios"""

    def __init__(self, filename: str, config: dict):
        super().__init__()
        self.filename = filename
        self.config = config
        self._init_services()
        self._init_ui()
        self._connect_events()

    def _init_services(self):
        """Inicializa servicios y presentadores"""
        self.data_service = DataService(
            FitsRepository(),
            ConfigRepository()
        )
        self.event_service = EventService()
        self.cube_presenter = CubePresenter()
        self.spectrum_presenter = SpectrumPresenter()

        # Cargar datos
        self.cube_data = self.data_service.load_cube(self.filename)
        self.wavelength = self.cube_data.wavelength
        self.current_slice = self.cube_data.get_slice(0)

    def _init_ui(self):
        """Configura la interfaz gráfica principal"""
        self.setWindowTitle(f"ViewCube - {self.filename}")
        self.resize(1200, 800)

        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principal
        layout = QVBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)

        # Vista de spaxels
        self.spaxel_view = pg.GraphicsLayoutWidget()
        self.spaxel_plot = self.spaxel_view.addPlot(title="Cube Viewer")
        self.spaxel_image = pg.ImageItem()
        self.spaxel_plot.addItem(self.spaxel_image)

        # Vista de espectros
        self.spectrum_view = pg.GraphicsLayoutWidget()
        self.spectrum_plot = self.spectrum_view.addPlot(title="Spectrum Viewer")

        splitter.addWidget(self.spaxel_view)
        splitter.addWidget(self.spectrum_view)
        layout.addWidget(splitter)

        # Configurar visualización inicial
        self._update_spaxel_view()
        self._update_spectrum_view()

    def _connect_events(self):
        """Conecta eventos de UI a servicios"""
        # Eventos de ratón
        self.spaxel_plot.scene().sigMouseMoved.connect(self._handle_mouse_move)
        self.spaxel_plot.scene().sigMouseClicked.connect(self._handle_mouse_click)

        # Eventos de teclado
        self.event_service.register_handler("key_press", self._handle_key_press)

    def _update_spaxel_view(self):
        """Actualiza la vista de spaxels usando el presentador"""
        img = self.cube_presenter.present_slice(
            self.current_slice,
            colormap=self.config.get('colormap', 'viridis'),
            scale=self.config.get('scale', 'linear')
        )
        self.spaxel_image.setImage(img.T)

    def _update_spectrum_view(self, x: int = None, y: int = None):
        """Actualiza la vista de espectros"""
        self.spectrum_plot.clear()

        if x is not None and y is not None:
            spectrum = self.cube_data.get_spectrum(x, y)
            self.spectrum_presenter.present_spectrum(
                self.wavelength,
                spectrum,
                title=f"Spectrum ({x}, {y})"
            )

    def _handle_mouse_move(self, pos):
        """Maneja movimiento del ratón en spaxel viewer"""
        if self.spaxel_image.image is not None:
            x, y = int(pos.x()), int(pos.y())
            if 0 <= x < self.current_slice.shape[1] and 0 <= y < self.current_slice.shape[0]:
                self._update_spectrum_view(x, y)

    def _handle_mouse_click(self, event):
        """Maneja clics en spaxel viewer"""
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            x = int(pos.x())
            y = int(pos.y())
            self.event_service.emit_event("spaxel_select", {"x": x, "y": y})

    def _handle_key_press(self, event):
        """Maneja eventos de teclado"""
        key = event.key()
        # Navegación de slices
        if key == Qt.Key_Left:
            self._change_slice(-1)
        elif key == Qt.Key_Right:
            self._change_slice(1)

    def _change_slice(self, direction: int):
        """Cambia el slice actual del cubo"""
        current_idx = self.cube_data.current_slice
        new_idx = max(0, min(current_idx + direction, len(self.wavelength) - 1))
        if new_idx != current_idx:
            self.cube_data.current_slice = new_idx
            self.current_slice = self.cube_data.get_slice(new_idx)
            self._update_spaxel_view()