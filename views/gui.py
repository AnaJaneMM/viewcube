import os
import platform
import sys

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QTabWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QComboBox,
                             QFrame, QFileDialog, QAction, QGroupBox, QMessageBox, QDoubleSpinBox, QMenu, QSizePolicy,
                             QLineEdit, QCheckBox)
from astropy.io import fits
from config import strings, styles
from viewcube import cubeviewer_optimized_short as cv
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
        self.chart.setFixedWidth(400)
        self.chart.setFixedHeight(400)
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
        try:
            super().__init__()
            
            # Configurar la ventana principal
            self.setWindowTitle("ViewCube")
            self.setup_window_size()
            
            # Crear widget central
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            
            # Crear layout principal
            self.main_layout = QVBoxLayout(self.central_widget)
            self.main_layout.setContentsMargins(10, 10, 10, 10)
            self.main_layout.setSpacing(10)
            
            # Inicializar variables
            self.cube_viewer = None
            self.current_file = None
            self.current_table = None
            self.current_comparison = None
            
            # Configurar la interfaz
            self.setup_menu()
            
            # Crear layout para el panel de configuración
            config_layout = QHBoxLayout()
            self.setup_config_panel(config_layout)
            self.main_layout.addLayout(config_layout)
            
            # Crear layout para el área de trabajo
            workspace_layout = QVBoxLayout()
            self.setup_workspace_panel(workspace_layout)
            self.main_layout.addLayout(workspace_layout)
            
            # Conectar acciones del menú
            self.connect_menu_actions()
            
            # Mostrar la ventana
            self.show()
            
        except Exception as e:
            QMessageBox.critical(None, "Error de Inicialización", 
                               f"Error al inicializar la ventana principal: {str(e)}")
            raise

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
        self.data_ext_spin.setToolTip(strings.DATA_EXTENSION_TOOL_TIP)
        self.data_ext_spin.setRange(0, 100)
        data_ext_layout.addWidget(self.data_ext_spin)
        
        # ERROR extension
        error_ext_layout = QHBoxLayout()
        error_ext_layout.addWidget(QLabel(strings.ERROR_EXTENSION))
        self.error_ext_spin = QSpinBox()
        self.error_ext_spin.setRange(0, 100)
        self.error_ext_spin.setToolTip(strings.ERROR_EXTENSION_TOOL_TIP)
        self.error_ext_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        error_ext_layout.addWidget(self.error_ext_spin)
        
        # FLAG/MASK extension
        flag_ext_layout = QHBoxLayout()
        flag_ext_layout.addWidget(QLabel(strings.FLAG_EXTENSION))
        self.flag_ext_spin = QSpinBox()
        self.flag_ext_spin.setRange(0, 100)
        self.flag_ext_spin.setToolTip(strings.FLAG_EXTENSION_TOOL_TIP)
        self.flag_ext_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        flag_ext_layout.addWidget(self.flag_ext_spin)
        
        # HEADER extension
        header_ext_layout = QHBoxLayout()
        header_ext_layout.addWidget(QLabel(strings.HEADER_EXTENSION))
        self.header_ext_spin = QSpinBox()
        self.header_ext_spin.setRange(0, 100)
        self.header_ext_spin.setValue(0)
        self.header_ext_spin.setToolTip(strings.HEADER_EXTENSION_TOOL_TIP)
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
        self.angle_rotation_value.setToolTip(strings.ROTATION_ANGLE_TOOL_TIP)
        angle_layout.addWidget(QLabel(strings.ROTATION_ANGLE))
        angle_layout.addWidget(self.angle_rotation_value)

        # Sensitivity
        sensitivity_layout = QHBoxLayout()
        self.sensitivity_value = QComboBox() #  store_false event
        self.sensitivity_value.addItems(["False", "True"])
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
        self.fo_factor_spin.setToolTip(strings.ORIGINAL_MULTIPLICATIVE_FACTOR_TOOL_TIP)
        fo_factor_layout.addWidget(self.fo_factor_spin)
        
        # comparison file multiplicative factor
        fc_factor_layout = QHBoxLayout()
        fc_factor_layout.addWidget(QLabel(strings.COMPARISON_MULTIPLICATIVE_FACTOR))
        self.fc_factor_spin = QDoubleSpinBox()
        self.fc_factor_spin.setRange(0, 100)
        self.fc_factor_spin.setSingleStep(0.1)
        self.fc_factor_spin.setValue(1.0)
        self.fc_factor_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.fc_factor_spin.setToolTip(strings.COMPARISON_MULTIPLICATIVE_FACTOR_TOOL_TIP)
        fc_factor_layout.addWidget(self.fc_factor_spin)
        
        # IVAR to error checkbox
        ivar_layout = QHBoxLayout()
        ivar_layout.addWidget(QLabel(strings.IVAR_TO_ERROR))
        self.ivar_combo = QComboBox()
        self.ivar_combo.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.ivar_combo.setToolTip(strings.IVAR_TO_ERROR_TOOL_TIP)
        self.ivar_combo.addItems(["False", "True"])
        ivar_layout.addWidget(self.ivar_combo)

        # X,Y instead of masked arrays
        m_layout = QHBoxLayout()
        m_layout.addWidget(QLabel('X/Y instead of masked arrays'))
        self.m_checkbox = QCheckBox()
        self.m_checkbox.setChecked(False)
        self.m_checkbox.setToolTip('Do NOT use masked arrays for flagged values')
        self.m_checkbox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        m_layout.addWidget(self.m_checkbox)

        # External position table
        p_layout = QHBoxLayout()
        p_layout.addWidget(QLabel("External position table for RSS Viewer"))
        self.p_lineedit = QLineEdit()
        self.p_lineedit.setPlaceholderText("Ruta a la tabla de posiciones")
        p_layout.addWidget(self.p_lineedit)

        # Dimensión espectral
        s_layout = QHBoxLayout()
        s_layout.addWidget(QLabel("Spectral dimension"))
        self.s_spinbox = QSpinBox()
        self.s_spinbox.setMinimum(0)
        self.s_spinbox.setMaximum(10000)
        self.s_spinbox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        s_layout.addWidget(self.s_spinbox)

        # Plot style
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Plot style (comma-separated)"))
        self.y_lineedit = QLineEdit()
        self.y_lineedit.setPlaceholderText("dark_background, seaborn-ticks")
        self.y_lineedit.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        y_layout.addWidget(self.y_lineedit)

        # Print version
        v_layout = QHBoxLayout()
        v_layout.addWidget(QLabel("Print version"))
        self.v_checkbox = QCheckBox()
        self.v_checkbox.setChecked(False)
        v_layout.addWidget(self.v_checkbox)

        # HDU number
        w_layout = QHBoxLayout()
        w_layout.addWidget(QLabel("HDU number extension for the wavelength array"))
        self.w_lineedit = QLineEdit()
        self.w_lineedit.setPlaceholderText("Número de extensión HDU")
        w_layout.addWidget(self.w_lineedit)

        # config file
        config_file_layout = QHBoxLayout()
        config_file_layout.addWidget(QLabel("Write config file"))
        self.config_file_checkbox = QCheckBox()
        self.config_file_checkbox.setChecked(False)
        config_file_layout.addWidget(self.config_file_checkbox)

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
        layout.addLayout(m_layout)
        layout.addLayout(p_layout)
        layout.addLayout(s_layout)
        layout.addLayout(y_layout)
        layout.addLayout(v_layout)
        layout.addLayout(w_layout)
        layout.addLayout(config_file_layout)
        layout.addLayout(load_layout)

    def setup_workspace_panel(self, layout):
        """Configura el panel de trabajo con los gráficos"""
        try:
            # Crear un layout horizontal para los gráficos
            graphics_layout = QHBoxLayout()
            graphics_layout.setContentsMargins(0, 0, 0, 0)
            graphics_layout.setSpacing(5)
            
            # Configurar el widget de spaxels
            self.spaxel_widget = pg.GraphicsLayoutWidget()
            self.spaxel_widget.setObjectName("spaxel_widget")
            self.spaxel_widget.setMinimumSize(600, 600)
            self.spaxel_plot = self.spaxel_widget.addPlot()
            self.spaxel_plot.showGrid(x=True, y=True)
            
            # Configurar el widget de espectros
            self.spectrum_widget = pg.GraphicsLayoutWidget()
            self.spectrum_widget.setObjectName("spectrum_widget")
            self.spectrum_widget.setMinimumSize(600, 600)
            self.spectrum_plot = self.spectrum_widget.addPlot()
            self.spectrum_plot.showGrid(x=True, y=True)
            
            # Añadir widgets al layout con stretch para que ocupen el mismo espacio
            graphics_layout.addWidget(self.spaxel_widget, stretch=1)
            graphics_layout.addWidget(self.spectrum_widget, stretch=1)
            
            # Añadir el layout de gráficos al layout principal
            layout.addLayout(graphics_layout)
            
            # Guardar referencias
            self.plots_layout = graphics_layout
            
        except Exception as e:
            QMessageBox.critical(None, "Error", 
                               f"Error al inicializar el panel de trabajo: {str(e)}")
            raise

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
        self.cube_viewer = None
        self.current_file = None
        self.current_table = None
        self.current_comparison = None

        # Limpiar gráficos
        if self.cube_viewer:
            self.cube_viewer.spaxel_widget.clear_graph()
            self.cube_viewer.spectrum_widget.clear_graph()

        # Resetear título
        self.setWindowTitle("ViewCube")

    def show_window_manager(self):
        """Muestra el administrador de ventanas"""
        if self.cube_viewer:
            self.cube_viewer.WindowManager()

    def show_lambda_limits(self):
        """Muestra el diálogo de límites lambda"""
        if self.cube_viewer:
            self.cube_viewer.show_lambda_limits_dialog()
        else:
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def on_search_fits(self):
        """Method event linked to the search of a FITS file."""
        file_path, _ = QFileDialog.getOpenFileName(self, strings.SEARCH_FITS_MSG, "", strings.SEARCH_FITS_FILTER)
        if file_path:
            self.current_file = file_path
            self.file_name_box.setText(os.path.basename(file_path))
            self.btn_load.setEnabled(True)
    
    def on_load_clicked(self):
        """Maneja el evento de clic en el botón de carga"""
        if hasattr(self, 'current_file'):
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
                self.cube_viewer = cv.CubeViewer(
                    name_fits=self.current_file,
                    ptable=self.position_table_box.text(),
                    fitscom=self.comp_file_name_box.text(),
                    **kwargs
                )
                
                # Limpiar los layouts existentes
                self.plots_layout.clear()
                
                # Crear el adaptador para PyQtGraph
                self.cube_viewer_adapter = CubeViewerAdapter(self.cube_viewer)
                
                # Añadir los nuevos widgets a los layouts de GraphicsLayoutWidget
                self.spaxel_widget.addItem(self.cube_viewer_adapter.spaxel_widget.plotItem)
                self.spectrum_widget.addItem(self.cube_viewer_adapter.spectrum_widget.plotItem)
                
                # Actualizar título de la ventana
                self.setWindowTitle(f"ViewCube - {os.path.basename(self.current_file)}")
                
                # Mostrar información del archivo
                try:
                    info = [
                        fits_file_info(self.current_file),
                        "",
                        "Información del cubo:",
                        self.cube_viewer.get_info()
                    ]
                    QMessageBox.information(self, "Información del archivo", "\n".join(info))
                except Exception as e:
                    print(f"Error mostrando información: {e}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al cargar el archivo:\n{str(e)}")
                self.current_file = None
                self.cube_viewer = None
                self.cube_viewer_adapter = None
                self.setWindowTitle("ViewCube")
        else:
            QMessageBox.warning(self, "Error", "Por favor, seleccione un archivo FITS primero")

    def on_search_table(self):
        file_path, _ = QFileDialog.getOpenFileName(self, strings.SEARCH_POSITION_TABLE_MSG, "", strings.SEARCH_ALL_FILES)
        if file_path:
            try:
                self.current_table = file_path     # load position table

                # update with new table if cube is loading
                if self.cube_viewer:
                    self.cube_viewer.ptable = self.current_table
                    self.update_visualizations()

                QMessageBox.information(self, strings.GENERIC_SUCCESS_TITLE, strings.POSITION_TABLE_LOADED)
            except Exception as e:
                QMessageBox.critical(self, strings.GENERIC_ERROR_TITLE, strings.ERROR_LOADING_POS_TABLE + str(e))

    def on_search_comparison(self):
        """Method event linked to the search of a comparison FITS file."""
        file_path, _ = QFileDialog.getOpenFileName(self, strings.SEARCH_COMPARISSON_FITS_MSG, "", strings.SEARCH_FITS_FILTER)
        if file_path:
            try:
                self.current_comparison = file_path

                # update cube if another cube is already loaded
                if self.cube_viewer:
                    self.cube_viewer.fitscom = self.current_comparison
                    self.comp_file_name_box.setText(os.path.basename(file_path))
                    print(os.path.basename(file_path))
                    self.load_fits_file(self.cube_viewer.name_fits)
                
                QMessageBox.information(self, strings.GENERIC_SUCCESS_TITLE, strings.COMPARISON_FILE_LOADED)
            except Exception as e:
                QMessageBox.critical(self, strings.GENERIC_ERROR_TITLE, strings.ERROR_LOADING_COMPARISON_FITS_FILE + str(e))
                self.current_comparison = None

    def update_cube_parameters(self):
        """Actualiza los parámetros del cubo cuando cambian los controles"""
        if self.cube_viewer:
            try:
                # Actualizar extensiones
                self.cube_viewer.kwargs['exdata'] = self.data_ext_spin.value() if self.data_ext_spin.value() > 0 else None
                self.cube_viewer.kwargs['exhdr'] = self.header_ext_spin.value()
                self.cube_viewer.kwargs['exerror'] = self.error_ext_spin.value() if self.error_ext_spin.value() > 0 else None
                self.cube_viewer.kwargs['exflag'] = self.flag_ext_spin.value() if self.flag_ext_spin.value() > 0 else None

                # Actualizar factores de multiplicación
                self.cube_viewer.kwargs['fo'] = self.fo_factor_spin.value()
                self.cube_viewer.kwargs['fc'] = self.fc_factor_spin.value()

                # Actualizar otros parámetros
                self.cube_viewer.kwargs['ivar'] = self.ivar_combo.currentText() == "True"

                # Recargar el cubo con los nuevos parámetros
                self.load_fits_file(self.cube_viewer.name_fits)
            except Exception as e:
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
            self.cube_viewer = cv.CubeViewer(
                name_fits=self.current_file,
                ptable=self.position_table_box.text(),
                fitscom=self.comp_file_name_box.text(),
                **kwargs
            )
            
            # Limpiar los layouts existentes
            self.plots_layout.clear()
            
            # Crear el adaptador para PyQtGraph
            self.cube_viewer_adapter = CubeViewerAdapter(self.cube_viewer)
            
            # Añadir los nuevos widgets a los layouts de GraphicsLayoutWidget
            self.spaxel_widget.addItem(self.cube_viewer_adapter.spaxel_widget.plotItem)
            self.spectrum_widget.addItem(self.cube_viewer_adapter.spectrum_widget.plotItem)
            
            # Actualizar título de la ventana
            self.setWindowTitle(f"ViewCube - {os.path.basename(self.current_file)}")
            
            # Mostrar información del archivo
            try:
                info = [
                    fits_file_info(self.current_file),
                    "",
                    "Información del cubo:",
                    self.cube_viewer.get_info()
                ]
                QMessageBox.information(self, "Información del archivo", "\n".join(info))
            except Exception as e:
                print(f"Error mostrando información: {e}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar el archivo:\n{str(e)}")
            self.current_file = None
            self.cube_viewer = None
            self.cube_viewer_adapter = None
            self.setWindowTitle("ViewCube")
    else:
        QMessageBox.warning(self, "Error", "Por favor, seleccione un archivo FITS primero")

def on_search_table(self):
    file_path, _ = QFileDialog.getOpenFileName(self, strings.SEARCH_POSITION_TABLE_MSG, "", strings.SEARCH_ALL_FILES)
    if file_path:
        try:
            self.current_table = file_path     # load position table

            # update with new table if cube is loading
            if self.cube_viewer:
                self.cube_viewer.ptable = self.current_table
                self.update_visualizations()

            QMessageBox.information(self, strings.GENERIC_SUCCESS_TITLE, strings.POSITION_TABLE_LOADED)
        except Exception as e:
            QMessageBox.critical(self, strings.GENERIC_ERROR_TITLE, strings.ERROR_LOADING_POS_TABLE + str(e))

def on_search_comparison(self):
    """Method event linked to the search of a comparison FITS file."""
    file_path, _ = QFileDialog.getOpenFileName(self, strings.SEARCH_COMPARISSON_FITS_MSG, "", strings.SEARCH_FITS_FILTER)
    if file_path:
        try:
            self.current_comparison = file_path
            QMessageBox.warning(self, "Error",
                              "Para ver residuos, primero debe cargar un archivo de comparación")

    def fit_spectrum(self):
        """Muestra la interfaz de ajuste de espectro"""
        if self.cube_viewer:
            self.cube_viewer.FitSpec(None)


def main():
    try:
        # Crear la aplicación
        app = QApplication(sys.argv)
        
        # Configurar el estilo
        app.setStyle('Fusion')
        
        # Configurar paleta de colores
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        app.setPalette(palette)
        
        # Configurar pyqtgraph
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True)
        
        # Crear y mostrar la ventana principal
        window = MainWindow()
        window.show()
        
        # Iniciar el bucle de eventos
        sys.exit(app.exec_())
        
    except Exception as e:
        # Mostrar mensaje de error
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("Error Fatal")
        error_msg.setText("Error al iniciar la aplicación")
        error_msg.setInformativeText(str(e))
        error_msg.exec_()
        return 1


if __name__ == '__main__':
    main() 