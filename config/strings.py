"""Text configurations for GUI"""
import viewcube.version as version

# General GUI text
WINDOW_TITLE = "ViewCube " + version.__version__
DEFAULT_TAB_NAME = "Workspace Tab"
SPECTRAL_FRAME_NAME = "Spectral Viewer"
SPAXEL_FRAME_NAME = "Spaxel Viewer"

# Configuration for explorer
SEARCH_FITS_MSG = 'Seleccionar fichero FITS'
SEARCH_COMPARISSON_FITS_MSG = 'Seleccionar fichero FITS de comparación'
SEARCH_POSITION_TABLE_MSG = 'Seleccionar tabla de posiciones'
SEARCH_FITS_FILTER = 'FITS files (*.fits *.fit)'
SEARCH_ALL_FILES = 'All Files (*.*)'

# Menu bar text
FILE_MENU = "Archivo"
TOOLS_MENU = "Herramientas"
MANAGER_MENU = "Administrador"

# Menu items
OPEN_FITS = "Abrir archivo FITS"
NEW_WORKSPACE = "Crear nuevo espacio de trabajo"
SAVE_SPECTRUM = "Guardar espectro"
WINDOW_MANAGER = "Administrador de ventanas"
LAMBDA_LIMITS = "Límites Lambda"
SONIFICATION = "Sonificación"
VIEW_RESIDUALS = "Ver residuos"
FIT_SPECTRUM = "Ajustar espectro"

# Frames and layout
CONFIGURATION_FRAME = "Configuración"
PLOTS_FRAME = "Gráficas"

# Configuration widgets
LABEL_FILE_PATH = "Fichero FITS principal"
GENERIC_SEARCH_BUTTON = 'Buscar'
BUTTON_SELECT_FILES = "Seleccionar archivos"
LABEL_NO_FILE = "Archivo actual: ninguno"
EXTERNAL_POSITION_TABLE = 'Tabla de posiciones externa para RSS Viewer'
ANGLE_ROTATION = 'Angle to rotate the position table (only RSS)'
COMPARISON_FITS_FILE = 'Fichero FITS de comparación'
DATA_EXTENSION = 'DATA extension (default: None)'
ERROR_EXTENSION = 'ERROR extension (default: None)'
FLAG_EXTENSION = 'FLAG/MASK extension (default: None)'
HEADER_EXTENSION = 'HEADER extension (default: 0)'
SENSITIVITY = 'Sensitivity curve'
SENSITIVITY_TOOL_TIP = 'Do NOT apply sensitivity curve (if HDU is available)'
ORIGINAL_MULTIPLICATIVE_FACTOR = 'Original multiplicative factor'
ORIGINAL_MULTIPLICATIVE_FACTOR_HELP = 'Multiplicative factor for the original file'
COMPARISON_MULTIPLICATIVE_FACTOR = 'Comparison multiplicative factor'
COMPARISON_MULTIPLICATIVE_FACTOR_HELP = 'Multiplicative factor for the comparison file'
IVAR_TO_ERROR = 'IVAR to Error'
IVAR_TO_ERROR_HELP = 'Conversion from IVAR to error'

# Messages
ERROR_NO_DATA = "No se pudieron leer datos del archivo FITS"
ERROR_NOT_3D = "¡Este archivo FITS no es un cubo 3D!"
ERROR_LOADING = "Error al cargar el archivo: {}"
ERROR_NO_COMPARISON = "Para ver residuos, primero debe cargar un archivo de comparación"

# Plot labels
SPAXEL_TITLE = "Spaxel (Canal {})"
SPECTRUM_TITLE = "Espectro en ({}, {})"
SPECTRUM_SELECT = "Seleccione un punto en el spaxel"
SPECTRUM_REGION = "Espectro promedio de región ({},{}) a ({},{})"
LABEL_CHANNEL = "Canal"
LABEL_INTENSITY = "Intensidad"

# Information messages
GENERIC_SUCCESS_TITLE = 'Exito'
POSITION_TABLE_LOADED = 'Tabla de posiciones cargada correctamente'
COMPARISON_FILE_LOADED = 'Archivo de comparación cargado correctamente'

# Error messages
GENERIC_ERROR_TITLE = 'Error'
ERROR_LOADING_POS_TABLE = 'Error al cargar la tabla de posiciones: '
ERROR_LOADING_COMPARISON_FITS_FILE = 'Error al cargar el fichero FITS de comparación: '
