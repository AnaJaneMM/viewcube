import os
import platform
import sys

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QTabWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QComboBox,
                             QFrame, QFileDialog, QAction, QGroupBox, QMessageBox, QDoubleSpinBox, QMenu, QSizePolicy,
                             QLineEdit)
from astropy.io import fits
from config import strings, styles
from viewcube import cubeviewer as cv
from viewcube.qt_adapter import CubeViewerAdapter

# Configuración global de pyqtgraph
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

def fits_file_info(file_path):
    """
    Shows a table with the info of the FITS file. The headers of the table are: 'No.', 'Name', 'Ver', 'Type',
    'Cards', 'Dimensions' and 'Format'.

    :param file_path: Path to the FITS file
    :return: All info in String format
    """
    info = []
    with fits.open(file_path) as hdul:
        info.append("Extensiones:")
        for i, hdu in enumerate(hdul):
            shape_str = str(hdu.data.shape) if hasattr(hdu, 'data') and hdu.data is not None else 'No data'
            info.append(f"{i}: {hdu.__class__.__name__}, shape={shape_str}")
    return "\n".join(info)


def file_extensions(file_path):
    """
    :param file_path:  Path to the FITS file
    :return:  Number of extensions in the FITS file
    """
    return len(fits.open(file_path))


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Crear el gráfico
        self.chart = pg.PlotWidget()
        self.chart.setBackgroundVisible(False)
        
        # Añadir la vista al layout
        self.layout.addWidget(self.chart)
        
        # Series de datos
        self.series = None
        
    def clear_graph(self):
        """Limpia el gráfico"""
        self.chart.clear()
        self.series = None
        
    def plot(self, x, y, color=Qt.blue, clear=True):
        """Dibuja una serie de datos"""
        if clear:
            self.clear_graph()
            
        # Crear nueva serie
        self.series = pg.PlotDataItem(x=x, y=y, pen=pg.mkPen(color))
        
        # Añadir serie al gráfico
        self.chart.addItem(self.series)
        
    def set_labels(self, title="", x_label="", y_label=""):
        """Establece las etiquetas del gráfico"""
        self.chart.setTitle(title)
        self.chart.setLabel('left', y_label)
        self.chart.setLabel('bottom', x_label)


class SpaxelWidget(PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scatter = pg.ScatterPlotItem()
        self.scatter.setData(x=[], y=[], pen=None, symbol='o', size=10, symbolBrush=(255, 0, 0))
        self.chart.addItem(self.scatter)
        
    def plot_spaxel(self, data, clear=True):
        """Dibuja el spaxel"""
        if clear:
            self.clear_graph()
            
        # Crear serie para la imagen
        self.series = pg.PlotDataItem(x=np.arange(data.shape[1]), y=data[0], pen=None)
        
        # Añadir serie al gráfico
        self.chart.addItem(self.series)
        
        # Actualizar rangos
        self.chart.setRange(xRange=(0, data.shape[1]), yRange=(0, data.shape[0]))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.btn_search_comp_fits = None
        self.comp_file_name_box = None
        self.btn_search_table = None
        self.position_table_box = None
        self.file_name_box = None
        self.btn_search_fits = None
        self.angle_rotation_value = None
        self.cube = None
        self.cube_adapter = None
        self.data = None
        self.comparison_cube = None
        self.position_table = None

        # Initial window configuration
        self.setWindowTitle(strings.WINDOW_TITLE)
        self.setup_window_size()

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)  # Cambiado a horizontal

        # Menu
        self.setup_menu()

        # Configuration's frame
        config_group = QGroupBox(strings.CONFIGURATION_FRAME)
        config_layout = QVBoxLayout(config_group)
        self.setup_config_panel(config_layout)

        # Plot's frame
        plots_group = QGroupBox(strings.PLOTS_FRAME)
        workspace_layout = QVBoxLayout(plots_group)
        self.setup_workspace_panel(workspace_layout)

        # Add frames to main layout
        main_layout.addWidget(config_group, 1)
        main_layout.addWidget(plots_group, 3)

        # Menu actions
        self.connect_menu_actions()

        # Mostrar la ventana maximizada
        self.showMaximized()

    def setup_window_size(self):
        if platform.system() == "Windows":
            self.showMaximized()
        else:
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(0, 0, screen.width(), screen.height())

    def setup_menu(self):
        menubar = self.menuBar()

        # Menú Archivo
        file_menu = menubar.addMenu(strings.FILE_MENU)
        file_menu.setObjectName(strings.FILE_MENU)

        open_action = QAction(strings.OPEN_FITS, self)
        open_action.setShortcut('Ctrl+O')
        file_menu.addAction(open_action)

        new_workspace_action = QAction(strings.NEW_WORKSPACE, self)
        new_workspace_action.setShortcut('Ctrl+N')
        file_menu.addAction(new_workspace_action)

        save_action = QAction(strings.SAVE_SPECTRUM, self)
        save_action.setShortcut('Ctrl+S')
        file_menu.addAction(save_action)

        # Menú Administrador
        manager_menu = menubar.addMenu(strings.MANAGER_MENU)
        manager_menu.setObjectName(strings.MANAGER_MENU)

        window_manager_action = QAction(strings.WINDOW_MANAGER, self)
        window_manager_action.setShortcut('Ctrl+W')
        manager_menu.addAction(window_manager_action)

        lambda_limits_action = QAction(strings.LAMBDA_LIMITS, self)
        lambda_limits_action.setShortcut('Ctrl+L')
        manager_menu.addAction(lambda_limits_action)

        # Menú Herramientas
        tools_menu = menubar.addMenu(strings.TOOLS_MENU)
        tools_menu.setObjectName(strings.TOOLS_MENU)

        sonification_action = QAction(strings.SONIFICATION, self)
        sonification_action.setShortcut('Ctrl+M')
        tools_menu.addAction(sonification_action)

        residual_action = QAction(strings.VIEW_RESIDUALS, self)
        residual_action.setShortcut('Ctrl+R')
        tools_menu.addAction(residual_action)

        fit_action = QAction(strings.FIT_SPECTRUM, self)
        fit_action.setShortcut('Ctrl+F')
        tools_menu.addAction(fit_action)

    def setup_config_panel(self, layout):
        # FITS File Path
        file_group_box = QGroupBox()
        file_group_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        #file_group_box.setStyleSheet(styles.GROUP_BOX_NO_BORDER)
        file_layout = QVBoxLayout(file_group_box)
        file_line_btn_layout = QHBoxLayout()
        self.file_name_box = QLineEdit()
        self.btn_search_fits = QPushButton(strings.GENERIC_SEARCH_BUTTON)
        self.btn_search_fits.clicked.connect(self.on_search_fits)
        self.btn_search_fits.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        file_layout.addWidget(QLabel(strings.LABEL_FILE_PATH))
        file_line_btn_layout.addWidget(self.file_name_box)
        file_line_btn_layout.addWidget(self.btn_search_fits)
        file_layout.addLayout(file_line_btn_layout)
        
        # Position table RSS only
        position_table_group_box = QGroupBox()
        position_table_group_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        #position_table_group_box.setStyleSheet(styles.GROUP_BOX_NO_BORDER)
        pos_table_layout = QVBoxLayout(position_table_group_box)
        pos_line_btn_layout = QHBoxLayout()
        self.position_table_box = QLineEdit()
        self.btn_search_table = QPushButton(strings.GENERIC_SEARCH_BUTTON)
        self.btn_search_table.clicked.connect(self.on_search_table)
        self.btn_search_table.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        pos_table_layout.addWidget(QLabel(strings.EXTERNAL_POSITION_TABLE))
        pos_line_btn_layout.addWidget(self.position_table_box)
        pos_line_btn_layout.addWidget(self.btn_search_table)
        pos_table_layout.addLayout(pos_line_btn_layout)

        
        # Fichero FITS de comparación
        comp_file_group_box = QGroupBox()
        comp_file_group_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        # comp_file_group_box.setStyleSheet(styles.GROUP_BOX_NO_BORDER)
        comp_file_layout = QVBoxLayout(comp_file_group_box)
        comp_file_line_btn_layout = QHBoxLayout()
        self.comp_file_name_box = QLineEdit()
        self.btn_search_comp_fits = QPushButton(strings.GENERIC_SEARCH_BUTTON)
        self.btn_search_comp_fits.clicked.connect(self.on_search_comparison)
        self.btn_search_comp_fits.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        comp_file_layout.addWidget(QLabel(strings.COMPARISON_FITS_FILE))
        comp_file_line_btn_layout.addWidget(self.comp_file_name_box)
        comp_file_line_btn_layout.addWidget(self.btn_search_comp_fits)
        comp_file_layout.addLayout(comp_file_line_btn_layout)
        # layout.addLayout(comp_file_layout) # no borrar


        #Extension group
        extension_group = QGroupBox()
        extension_layout = QVBoxLayout(extension_group)

        # DATA extension
        data_ext_layout = QHBoxLayout()
        data_ext_layout.addWidget(QLabel(strings.DATA_EXTENSION))
        self.data_ext_spin = QSpinBox()
        self.data_ext_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.data_ext_spin.setRange(0, 100)
        data_ext_layout.addWidget(self.data_ext_spin)
        
        # ERROR extension
        error_ext_layout = QHBoxLayout()
        error_ext_layout.addWidget(QLabel(strings.ERROR_EXTENSION))
        self.error_ext_spin = QSpinBox()
        self.error_ext_spin.setRange(0, 100)
        self.error_ext_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        error_ext_layout.addWidget(self.error_ext_spin)
        
        # FLAG/MASK extension
        flag_ext_layout = QHBoxLayout()
        flag_ext_layout.addWidget(QLabel(strings.FLAG_EXTENSION))
        self.flag_ext_spin = QSpinBox()
        self.flag_ext_spin.setRange(0, 100)
        self.flag_ext_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        flag_ext_layout.addWidget(self.flag_ext_spin)
        
        # HEADER extension
        header_ext_layout = QHBoxLayout()
        header_ext_layout.addWidget(QLabel(strings.HEADER_EXTENSION))
        self.header_ext_spin = QSpinBox()
        self.header_ext_spin.setRange(0, 100)
        self.header_ext_spin.setValue(0)
        self.header_ext_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        header_ext_layout.addWidget(self.header_ext_spin)

        # configure extension layout
        extension_layout.addLayout(data_ext_layout)
        extension_layout.addLayout(error_ext_layout)
        extension_layout.addLayout(flag_ext_layout)
        extension_layout.addLayout(header_ext_layout)

        # Rotation angle
        angle_layout = QHBoxLayout()
        self.angle_rotation_value = QDoubleSpinBox()
        self.angle_rotation_value.setRange(0, 360)
        self.angle_rotation_value.setValue(0)
        self.angle_rotation_value.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        angle_layout.addWidget(QLabel(strings.ANGLE_ROTATION))
        angle_layout.addWidget(self.angle_rotation_value)

        # Sensitivity
        sensitivity_layout = QHBoxLayout()
        self.sensitivity_value = QComboBox() #  store_false event
        self.sensitivity_value.addItem("True")
        self.sensitivity_value.addItem("False")
        self.sensitivity_value.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.sensitivity_value.setToolTip(strings.SENSITIVITY_TOOL_TIP)
        sensitivity_layout.addWidget(QLabel(strings.SENSITIVITY))
        sensitivity_layout.addWidget(self.sensitivity_value)

        # Original multiplicative factor
        fo_factor_layout = QHBoxLayout()
        fo_factor_layout.addWidget(QLabel(strings.ORIGINAL_MULTIPLICATIVE_FACTOR))
        self.fo_factor_spin = QDoubleSpinBox()
        self.fo_factor_spin.setRange(0, 100)
        self.fo_factor_spin.setSingleStep(0.1)
        self.fo_factor_spin.setValue(1.0)
        self.fo_factor_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.fo_factor_spin.setToolTip(strings.ORIGINAL_MULTIPLICATIVE_FACTOR_HELP)
        fo_factor_layout.addWidget(self.fo_factor_spin)
        
        # Factor de multiplicación comparación
        fc_factor_layout = QHBoxLayout()
        fc_factor_layout.addWidget(QLabel('Multiplicative factor for comparison file'))
        self.fc_factor_spin = QDoubleSpinBox()
        self.fc_factor_spin.setRange(0, 100)
        self.fc_factor_spin.setSingleStep(0.1)
        self.fc_factor_spin.setValue(1.0)
        fc_factor_layout.addWidget(self.fc_factor_spin)
        # layout.addLayout(fc_factor_layout) # no borrar
        
        # IVAR to error checkbox
        ivar_layout = QHBoxLayout()
        ivar_layout.addWidget(QLabel('Conversion from IVAR to error'))
        self.ivar_combo = QComboBox()
        self.ivar_combo.addItems(["False", "True"])
        ivar_layout.addWidget(self.ivar_combo)
        # layout.addLayout(ivar_layout) # no borrar
        
        # Botón de carga
        load_layout = QHBoxLayout()
        self.btn_load = QPushButton("Cargar")
        self.btn_load.clicked.connect(self.on_load_clicked)
        self.btn_load.setEnabled(False)  # Deshabilitado hasta que se seleccione un archivo
        load_layout.addWidget(self.btn_load)
        # layout.addLayout(load_layout) # no borrar

        # Adding the widgets in a certain order
        layout.addWidget(file_group_box)
        layout.addWidget(extension_group)
        layout.addLayout(angle_layout)
        layout.addWidget(comp_file_group_box)
        layout.addWidget(position_table_group_box)
        layout.addLayout(sensitivity_layout)
        layout.addLayout(fo_factor_layout) #
        layout.addLayout(fc_factor_layout)
        layout.addLayout(ivar_layout)
        layout.addLayout(load_layout)

    def setup_workspace_panel(self, layout):
        """Configura el panel de trabajo"""
        # Área de gráficos
        plots_layout = QHBoxLayout()
        
        # Frame izquierdo (Spaxel)
        spaxel_frame = QFrame()
        spaxel_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        spaxel_layout = QVBoxLayout(spaxel_frame)
        
        # Frame derecho (Espectro)
        spectrum_frame = QFrame()
        spectrum_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        spectrum_layout = QVBoxLayout(spectrum_frame)
        
        # Añadir los frames al layout principal
        plots_layout.addWidget(spaxel_frame)
        plots_layout.addWidget(spectrum_frame)
        
        # Añadir el layout de gráficos al layout principal
        layout.addLayout(plots_layout)
        
        # Configurar las proporciones del layout de gráficos
        plots_layout.setStretch(0, 1)  # Spaxel
        plots_layout.setStretch(1, 1)  # Spectrum
        
        # Guardar referencias a los layouts para usar más tarde
        self.spaxel_layout = spaxel_layout
        self.spectrum_layout = spectrum_layout

    def connect_menu_actions(self):
        """Conecta las acciones del menú con sus funciones correspondientes"""
        # Obtener las acciones del menú
        menubar = self.menuBar()
        file_menu = menubar.findChild(QMenu, 'Archivo')
        manager_menu = menubar.findChild(QMenu, 'Administrador')
        tools_menu = menubar.findChild(QMenu, 'Herramientas')

        # Conectar acciones del menú Archivo
        for action in file_menu.actions():
            if action.text() == 'Abrir archivo FITS':
                action.triggered.connect(self.on_search_fits)
            elif action.text() == 'Crear nuevo espacio de trabajo':
                action.triggered.connect(self.create_new_workspace)
            elif action.text() == 'Guardar espectro':
                action.triggered.connect(self.save_spectrum)

        # Conectar acciones del menú Administrador
        for action in manager_menu.actions():
            if action.text() == 'Administrador de ventanas':
                action.triggered.connect(self.show_window_manager)
            elif action.text() == 'Límites Lambda':
                action.triggered.connect(self.show_lambda_limits)

        # Conectar acciones del menú Herramientas
        for action in tools_menu.actions():
            if action.text() == 'Sonificación':
                action.triggered.connect(self.show_sonification)
            elif action.text() == 'Ver residuos':
                action.triggered.connect(self.show_residuals)
            elif action.text() == 'Ajustar espectro':
                action.triggered.connect(self.fit_spectrum)

    def create_new_workspace(self):
        """Crea un nuevo espacio de trabajo"""
        # Limpiar datos existentes
        self.cube = None
        self.cube_adapter = None
        self.data = None
        self.comparison_cube = None
        self.position_table = None

        # Limpiar gráficos
        if self.cube_adapter:
            self.cube_adapter.spaxel_widget.clear_graph()
            self.cube_adapter.spectrum_widget.clear_graph()

        # Resetear título
        self.setWindowTitle(strings.WINDOW_TITLE)

    def show_window_manager(self):
        """Muestra el administrador de ventanas"""
        if self.cube:
            self.cube.WindowManager()

    def show_lambda_limits(self):
        """Muestra el diálogo de límites lambda"""
        if self.cube_adapter:
            self.cube_adapter.show_lambda_limits_dialog()
        else:
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def on_search_fits(self):
        """Method event linked to the search of a FITS file."""
        file_path, _ = QFileDialog.getOpenFileName(self, strings.SEARCH_FITS_MSG, "", strings.SEARCH_FITS_FILTER)
        if file_path:
            self.selected_fits_file = file_path
            self.file_name_box.setText(os.path.basename(file_path))
            self.btn_load.setEnabled(True)
    
    def on_load_clicked(self):
        """Maneja el evento de clic en el botón de carga"""
        if hasattr(self, 'selected_fits_file'):
            try:
                # Crear el diccionario de kwargs con la configuración actual
                kwargs = {
                    'exdata': self.data_ext_spin.value() if self.data_ext_spin.value() > 0 else None,
                    'exhdr': self.header_ext_spin.value(),
                    'exerror': self.error_ext_spin.value() if self.error_ext_spin.value() > 0 else None,
                    'exflag': self.flag_ext_spin.value() if self.flag_ext_spin.value() > 0 else None,
                    'fo': self.fo_factor_spin.value(),
                    'fc': self.fc_factor_spin.value(),
                    'ivar': self.ivar_combo.currentText() == "True"
                }
                
                # Crear instancia de CubeViewer con los parámetros actuales
                self.cube = cv.CubeViewer(
                    name_fits=self.selected_fits_file,
                    ptable=self.position_table,
                    fitscom=self.comparison_cube,
                    **kwargs
                )
                
                # Limpiar los layouts existentes
                for i in reversed(range(self.spaxel_layout.count())): 
                    widget = self.spaxel_layout.itemAt(i).widget()
                    if widget is not None:
                        widget.setParent(None)
                
                for i in reversed(range(self.spectrum_layout.count())):
                    widget = self.spectrum_layout.itemAt(i).widget()
                    if widget is not None:
                        widget.setParent(None)
                
                # Crear el adaptador para PyQtGraph
                self.cube_adapter = CubeViewerAdapter(self.cube)
                
                # Añadir los nuevos widgets
                self.spaxel_layout.addWidget(self.cube_adapter.spaxel_widget)
                self.spectrum_layout.addWidget(self.cube_adapter.spectrum_widget)
                
                # Actualizar título de la ventana
                self.setWindowTitle(f"{strings.WINDOW_TITLE} - {os.path.basename(self.selected_fits_file)}")
                
                # Mostrar información del archivo
                try:
                    info = [
                        fits_file_info(self.selected_fits_file),
                        "",
                        "Información del cubo:",
                        self.cube.get_info()
                    ]
                    QMessageBox.information(self, "Información del archivo", "\n".join(info))
                except Exception as e:
                    print(f"Error mostrando información: {e}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al cargar el archivo:\n{str(e)}")
                self.data = None
                self.cube = None
                self.cube_adapter = None
                self.setWindowTitle(strings.WINDOW_TITLE)
        else:
            QMessageBox.warning(self, "Error", "Por favor, seleccione un archivo FITS primero")

    def on_search_table(self):
        file_path, _ = QFileDialog.getOpenFileName(self, strings.SEARCH_POSITION_TABLE_MSG, "", strings.SEARCH_ALL_FILES)
        if file_path:
            try:
                self.position_table = file_path     # load position table

                # update with new table if cube is loading
                if self.cube:
                    self.cube.ptable = self.position_table
                    self.update_visualizations()

                QMessageBox.information(self, strings.GENERIC_SUCCESS_TITLE, strings.POSITION_TABLE_LOADED)
            except Exception as e:
                QMessageBox.critical(self, strings.GENERIC_ERROR_TITLE, strings.ERROR_LOADING_POS_TABLE + str(e))

    def on_search_comparison(self):
        """Method event linked to the search of a comparison FITS file."""
        file_path, _ = QFileDialog.getOpenFileName(self, strings.SEARCH_COMPARISSON_FITS_MSG, "", strings.SEARCH_FITS_FILTER)
        if file_path:
            try:
                self.comparison_cube = file_path

                # update cube if another cube is already loaded
                if self.cube:
                    self.cube.fitscom = self.comparison_cube
                    self.comp_file_name_box.setText(os.path.basename(file_path))
                    print(os.path.basename(file_path))
                    self.load_fits_file(self.cube.name_fits)
                
                QMessageBox.information(self, strings.GENERIC_SUCCESS_TITLE, strings.COMPARISON_FILE_LOADED)
            except Exception as e:
                QMessageBox.critical(self, strings.GENERIC_ERROR_TITLE, strings.ERROR_LOADING_COMPARISON_FITS_FILE + str(e))
                self.comparison_cube = None

    def update_cube_parameters(self):
        """Actualiza los parámetros del cubo cuando cambian los controles"""
        if self.cube:
            try:
                # Actualizar extensiones
                self.cube.kwargs['exdata'] = self.data_ext_spin.value() if self.data_ext_spin.value() > 0 else None
                self.cube.kwargs['exhdr'] = self.header_ext_spin.value()
                self.cube.kwargs['exerror'] = self.error_ext_spin.value() if self.error_ext_spin.value() > 0 else None
                self.cube.kwargs['exflag'] = self.flag_ext_spin.value() if self.flag_ext_spin.value() > 0 else None

                # Actualizar factores de multiplicación
                self.cube.kwargs['fo'] = self.fo_factor_spin.value()
                self.cube.kwargs['fc'] = self.fc_factor_spin.value()

                # Actualizar otros parámetros
                self.cube.kwargs['ivar'] = self.ivar_combo.currentText() == "True"

                # Recargar el cubo con los nuevos parámetros
                self.load_fits_file(self.cube.name_fits)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error actualizando parámetros: {str(e)}")

    def load_fits_file(self, file_path):
        """Carga un archivo FITS y actualiza las visualizaciones"""
        try:
            # Crear instancia de CubeViewer con los parámetros de la interfaz
            self.cube = cv.CubeViewer(
                name_fits=file_path,
                ptable=self.position_table,
                fitscom=self.comparison_cube,
                exdata=self.data_ext_spin.value() if self.data_ext_spin.value() > 0 else None,
                exhdr=self.header_ext_spin.value(),
                exerror=self.error_ext_spin.value() if self.error_ext_spin.value() > 0 else None,
                exflag=self.flag_ext_spin.value() if self.flag_ext_spin.value() > 0 else None,
                ivar=self.ivar_combo.currentText() == "True"
            )
            
            # Limpiar los layouts existentes
            for i in reversed(range(self.spaxel_layout.count())): 
                widget = self.spaxel_layout.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
            
            for i in reversed(range(self.spectrum_layout.count())):
                widget = self.spectrum_layout.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
            
            # Crear el adaptador para PyQtGraph
            self.cube_adapter = CubeViewerAdapter(self.cube)
            
            # Añadir los nuevos widgets
            self.spaxel_layout.addWidget(self.cube_adapter.spaxel_widget)
            self.spectrum_layout.addWidget(self.cube_adapter.spectrum_widget)
            
            # Actualizar título de la ventana
            self.setWindowTitle(f"{strings.WINDOW_TITLE} - {os.path.basename(file_path)}")
            
            # Mostrar información del archivo
            try:
                info = [
                    fits_file_info(file_path),
                    "",
                    "Información del cubo:",
                    self.cube.get_info()
                ]
                QMessageBox.information(self, "Información del archivo", "\n".join(info))
            except Exception as e:
                print(f"Error mostrando información: {e}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar el archivo:\n{str(e)}")
            self.data = None
            self.cube = None
            self.cube_adapter = None
            self.setWindowTitle(strings.WINDOW_TITLE)

    def update_visualizations(self):
        """Actualiza las visualizaciones de spaxel y espectro"""
        if self.cube_adapter:
            self.cube_adapter.update_visualizations()

    def save_spectrum(self):
        """Guarda el espectro actual"""
        if self.cube:
            self.cube.SaveFile()

    def show_sonification(self):
        """Muestra la interfaz de sonificación"""
        if self.cube:
            self.cube.Sonification()

    def show_residuals(self):
        """Muestra el visor de residuos"""
        if self.cube_adapter and self.comparison_cube:
            self.cube_adapter.show_residuals()
        else:
            QMessageBox.warning(self, "Error",
                              "Para ver residuos, primero debe cargar un archivo de comparación")

    def fit_spectrum(self):
        """Muestra la interfaz de ajuste de espectro"""
        if self.cube:
            self.cube.FitSpec(None)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 