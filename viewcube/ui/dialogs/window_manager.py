from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal


class WindowManager(QDialog):
    windowSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Window Manager")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.resize(400, 300)

        self.layout = QVBoxLayout()
        self.window_list = QListWidget()

        # Botones
        btn_layout = QHBoxLayout()
        self.show_btn = QPushButton("Show")
        self.close_btn = QPushButton("Close")
        self.refresh_btn = QPushButton("Refresh")

        btn_layout.addWidget(self.show_btn)
        btn_layout.addWidget(self.close_btn)
        btn_layout.addWidget(self.refresh_btn)

        self.layout.addWidget(self.window_list)
        self.layout.addLayout(btn_layout)
        self.setLayout(self.layout)

        # Conexiones
        self.show_btn.clicked.connect(self._show_selected)
        self.close_btn.clicked.connect(self._close_selected)
        self.refresh_btn.clicked.connect(self.refresh_windows)
        self.window_list.itemDoubleClicked.connect(self._show_selected)

    def refresh_windows(self, windows):
        """Actualiza la lista de ventanas disponibles"""
        self.window_list.clear()
        for win in windows:
            self.window_list.addItem(win)

    def _show_selected(self):
        items = self.window_list.selectedItems()
        if items:
            self.windowSelected.emit(items[0].text())

    def _close_selected(self):
        items = self.window_list.selectedItems()
        if items:
            self.parent().close_window(items[0].text())
            self.refresh_windows(self.parent().get_window_list())