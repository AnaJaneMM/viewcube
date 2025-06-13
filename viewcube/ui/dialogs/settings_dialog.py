from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QDoubleSpinBox, QCheckBox,
                             QPushButton, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal


class SettingsDialog(QDialog):
    settingsChanged = pyqtSignal(dict)

    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ViewCube Settings")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.resize(500, 400)

        self.layout = QVBoxLayout()
        self._create_ui(current_settings)
        self.setLayout(self.layout)

    def _create_ui(self, settings):
        # Sección de Visualización
        vis_frame = QFrame()
        vis_frame.setFrameShape(QFrame.StyledPanel)
        vis_layout = QVBoxLayout()

        # Colormap
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
        self.cmap_combo.setCurrentText(settings.get('colormap', 'viridis'))
        vis_layout.addWidget(QLabel("Color Map:"))
        vis_layout.addWidget(self.cmap_combo)

        # Normalización
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(['linear', 'log', 'sqrt', 'power', 'asinh'])
        self.norm_combo.setCurrentText(settings.get('norm', 'sqrt'))
        vis_layout.addWidget(QLabel("Normalización:"))
        vis_layout.addWidget(self.norm_combo)

        # Límites dinámicos
        self.auto_range = QCheckBox("Auto Range")
        self.auto_range.setChecked(settings.get('auto_range', True))
        vis_layout.addWidget(self.auto_range)

        range_layout = QHBoxLayout()
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1e6, 1e6)
        self.min_spin.setValue(settings.get('vmin', 0))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1e6, 1e6)
        self.max_spin.setValue(settings.get('vmax', 1))
        range_layout.addWidget(QLabel("Min:"))
        range_layout.addWidget(self.min_spin)
        range_layout.addWidget(QLabel("Max:"))
        range_layout.addWidget(self.max_spin)
        vis_layout.addLayout(range_layout)

        vis_frame.setLayout(vis_layout)
        self.layout.addWidget(vis_frame)

        # Sección de Espectros
        spec_frame = QFrame()
        spec_frame.setFrameShape(QFrame.StyledPanel)
        spec_layout = QVBoxLayout()

        # Línea base
        self.baseline = QCheckBox("Remover línea base")
        self.baseline.setChecked(settings.get('baseline', False))
        spec_layout.addWidget(self.baseline)

        # Suavizado
        smooth_layout = QHBoxLayout()
        self.smooth_check = QCheckBox("Suavizado:")
        self.smooth_check.setChecked(settings.get('smooth_enabled', False))
        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.1, 10.0)
        self.smooth_spin.setValue(settings.get('smooth_factor', 2.0))
        smooth_layout.addWidget(self.smooth_check)
        smooth_layout.addWidget(self.smooth_spin)
        smooth_layout.addWidget(QLabel("σ"))
        spec_layout.addLayout(smooth_layout)

        spec_frame.setLayout(spec_layout)
        self.layout.addWidget(spec_frame)

        # Botones
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Aplicar")
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancelar")

        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        self.layout.addLayout(btn_layout)

        # Conexiones
        self.apply_btn.clicked.connect(self._apply_settings)
        self.ok_btn.clicked.connect(self._ok_pressed)
        self.cancel_btn.clicked.connect(self.reject)

    def _apply_settings(self):
        settings = {
            'colormap': self.cmap_combo.currentText(),
            'norm': self.norm_combo.currentText(),
            'auto_range': self.auto_range.isChecked(),
            'vmin': self.min_spin.value(),
            'vmax': self.max_spin.value(),
            'baseline': self.baseline.isChecked(),
            'smooth_enabled': self.smooth_check.isChecked(),
            'smooth_factor': self.smooth_spin.value()
        }
        self.settingsChanged.emit(settings)

    def _ok_pressed(self):
        self._apply_settings()
        self.accept()