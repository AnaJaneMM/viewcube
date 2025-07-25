import os
import platform
import sys
import logging as log
import colorlog
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QTabWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QComboBox,
                             QFrame, QFileDialog, QAction, QGroupBox, QMessageBox, QDoubleSpinBox, QMenu, QSizePolicy,
                             QLineEdit, QCheckBox)
from astropy.io import fits
from config import strings, styles, settings
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
        super().__init__()

        # log functionality
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            settings.FMT_LOG,
            log_colors={'DEBUG': 'cyan','INFO': 'green','WARNING': 'yellow','ERROR': 'red','CRITICAL': 'bold_red'}
        ))
        logger = colorlog.getLogger()
        logger.setLevel('DEBUG')
        logger.addHandler(handler)

        log.debug('Initializing MainWindow')

        # values
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
        main_layout = QHBoxLayout(central_widget)

        # Menu
        self.setup_menu()

        # Configuration's frame
        config_group = QGroupBox(strings.CONFIGURATION_FRAME)
        config_layout = QVBoxLayout(config_group)
        config_layout.setAlignment(Qt.AlignTop)
        self.setup_config_panel(config_layout)
        config_group.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

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

        log.debug('Configurando la barra de menú principal')

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

        flux_limits_action = QAction("Límites de flujo", self)
        flux_limits_action.setShortcut('Ctrl+F')
        manager_menu.addAction(flux_limits_action)

        redshift_action = QAction("Redshift", self)
        redshift_action.setShortcut('Ctrl+Z')
        manager_menu.addAction(redshift_action)

        rest_wave_action = QAction("Longitud de onda en reposo", self)
        rest_wave_action.setShortcut('Ctrl+R')
        manager_menu.addAction(rest_wave_action)

        # Menú Herramientas
        tools_menu = menubar.addMenu(strings.TOOLS_MENU)
        tools_menu.setObjectName(strings.TOOLS_MENU)

        sonification_action = QAction(strings.SONIFICATION, self)
        sonification_action.setShortcut('Ctrl+M')
        tools_menu.addAction(sonification_action)

        residual_action = QAction(strings.VIEW_RESIDUALS, self)
        residual_action.setShortcut('Ctrl+D')
        tools_menu.addAction(residual_action)

        fit_action = QAction(strings.FIT_SPECTRUM, self)
        fit_action.setShortcut('Ctrl+T')
        tools_menu.addAction(fit_action)

        synth_spec_action = QAction("Espectro sintético", self)
        synth_spec_action.setShortcut('Ctrl+Y')
        tools_menu.addAction(synth_spec_action)

        error_spec_action = QAction("Espectro de error", self)
        error_spec_action.setShortcut('Ctrl+E')
        tools_menu.addAction(error_spec_action)

        res_spec_action = QAction("Espectro residual", self)
        res_spec_action.setShortcut('Ctrl+X')
        tools_menu.addAction(res_spec_action)

        # Menú Visualización
        view_menu = menubar.addMenu("Visualización")
        view_menu.setObjectName("Visualización")

        change_filter_action = QAction("Cambiar filtro", self)
        change_filter_action.setShortcut('Ctrl+G')
        view_menu.addAction(change_filter_action)

        spectra_viewer_action = QAction("Visor de espectros", self)
        spectra_viewer_action.setShortcut('Ctrl+V')
        view_menu.addAction(spectra_viewer_action)

        change_spaxel_action = QAction("Cambiar vista de spaxel", self)
        change_spaxel_action.setShortcut('Ctrl+B')
        view_menu.addAction(change_spaxel_action)

        select_zone_action = QAction("Seleccionar zona", self)
        select_zone_action.setShortcut('Ctrl+A')
        view_menu.addAction(select_zone_action)

        # Menú Ayuda
        help_menu = menubar.addMenu("Ayuda")
        help_menu.setObjectName("Ayuda")

        about_action = QAction("Acerca de", self)
        about_action.setShortcut('F1')
        help_menu.addAction(about_action)

        # Conectar acciones
        open_action.triggered.connect(self.on_search_fits)
        new_workspace_action.triggered.connect(self.create_new_workspace)
        save_action.triggered.connect(self.save_spectrum)
        window_manager_action.triggered.connect(self.show_window_manager)
        lambda_limits_action.triggered.connect(self.show_lambda_limits)
        flux_limits_action.triggered.connect(self.show_flux_limits)
        redshift_action.triggered.connect(self.show_redshift)
        rest_wave_action.triggered.connect(self.show_rest_wave)
        sonification_action.triggered.connect(self.show_sonification)
        residual_action.triggered.connect(self.show_residuals)
        fit_action.triggered.connect(self.fit_spectrum)
        synth_spec_action.triggered.connect(self.show_synth_spec)
        error_spec_action.triggered.connect(self.show_error_spec)
        res_spec_action.triggered.connect(self.show_res_spec)
        change_filter_action.triggered.connect(self.change_filter)
        spectra_viewer_action.triggered.connect(self.show_spectra_viewer)
        change_spaxel_action.triggered.connect(self.change_spaxel_viewer)
        select_zone_action.triggered.connect(self.select_zone)
        about_action.triggered.connect(self.show_about)

        log.info('Barra de menú configurada exitosamente')

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

        log.debug('Configuration panel set up successfully')

    def setup_workspace_panel(self, layout):
        """Configura el panel de trabajo"""
        # Área de gráficos
        plots_layout = QHBoxLayout()
        
        # Frame izquierdo (Spaxel)
        ancho_spaxel = 710
        alto_spaxel = 600
        ancho_spectral = 800
        alto_spectral = 500
        spaxel_frame = QFrame()
        spaxel_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        spaxel_frame.setFixedSize(int(ancho_spaxel*0.70), int(alto_spaxel*0.70))
        spaxel_layout = QVBoxLayout(spaxel_frame)
        
        # Frame derecho (Espectro)
        spectrum_frame = QFrame()
        spectrum_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        spectrum_frame.setFixedSize(int(ancho_spectral*0.55), int(alto_spectral*0.70))
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
        log.debug('Workspace panel set up successfully')

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
        if self.cube:
            log.debug('Showing window manager')
            self.cube.WindowManager()
        else:
            log.warning('No cube loaded, cannot show window manager')

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

    def show_flux_limits(self):
        """Muestra el diálogo de límites de flujo"""
        if self.cube_adapter:
            log.debug('Mostrando diálogo de límites de flujo')
            self.cube_adapter.show_flux_limits_dialog()
        else:
            log.warning('No hay cubo cargado, no se pueden mostrar los límites de flujo')
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def show_redshift(self):
        """Muestra el diálogo de redshift"""
        if self.cube_adapter:
            log.debug('Mostrando diálogo de redshift')
            self.cube_adapter.show_redshift_dialog()
        else:
            log.warning('No hay cubo cargado, no se puede mostrar el diálogo de redshift')
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def show_rest_wave(self):
        """Muestra el diálogo de longitud de onda en reposo"""
        if self.cube_adapter:
            log.debug('Mostrando diálogo de longitud de onda en reposo')
            self.cube_adapter.show_rest_wave_dialog()
        else:
            log.warning('No hay cubo cargado, no se puede mostrar el diálogo de longitud de onda en reposo')
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def show_synth_spec(self):
        """Muestra el diálogo de espectro sintético"""
        if self.cube_adapter:
            log.debug('Mostrando diálogo de espectro sintético')
            self.cube_adapter.show_synth_spec_dialog()
        else:
            log.warning('No hay cubo cargado, no se puede mostrar el diálogo de espectro sintético')
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def show_error_spec(self):
        """Muestra el diálogo de espectro de error"""
        if self.cube_adapter:
            log.debug('Mostrando diálogo de espectro de error')
            self.cube_adapter.show_error_spec_dialog()
        else:
            log.warning('No hay cubo cargado, no se puede mostrar el diálogo de espectro de error')
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def show_res_spec(self):
        """Muestra el diálogo de espectro residual"""
        if self.cube_adapter:
            log.debug('Mostrando diálogo de espectro residual')
            self.cube_adapter.show_res_spec_dialog()
        else:
            log.warning('No hay cubo cargado, no se puede mostrar el diálogo de espectro residual')
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def change_filter(self):
        """Cambia el filtro actual"""
        if self.cube_adapter:
            log.debug('Cambiando filtro')
            self.cube_adapter.change_filter()
        else:
            log.warning('No hay cubo cargado, no se puede cambiar el filtro')
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def show_spectra_viewer(self):
        """Muestra el visor de espectros"""
        if self.cube_adapter:
            log.debug('Mostrando visor de espectros')
            self.cube_adapter.show_spectra_viewer()
        else:
            log.warning('No hay cubo cargado, no se puede mostrar el visor de espectros')
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def change_spaxel_viewer(self):
        """Cambia la vista de spaxel"""
        if self.cube_adapter:
            log.debug('Cambiando vista de spaxel')
            self.cube_adapter.change_spaxel_viewer()
        else:
            log.warning('No hay cubo cargado, no se puede cambiar la vista de spaxel')
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def select_zone(self):
        """Activa la selección de zona"""
        if self.cube_adapter:
            log.debug('Activando selección de zona')
            self.cube_adapter.select_zone()
        else:
            log.warning('No hay cubo cargado, no se puede activar la selección de zona')
            QMessageBox.warning(self, "Error", "Primero debe cargar un archivo FITS")

    def show_about(self):
        """Muestra el diálogo Acerca de"""
        log.debug('Mostrando diálogo Acerca de')
        QMessageBox.about(self, "Acerca de ViewCube",
                         "ViewCube - Visor de cubos de datos\n\n"
                         "Versión 1.0\n\n"
                         "Una herramienta para visualizar y analizar cubos de datos astronómicos.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 