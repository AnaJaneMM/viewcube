############################################################################
#                              VIEWCUBE                                    #
#                              PYTHON 3                                    #
#                                                                          #
# RGB@IAA ---> Last Change: 2024/10/10                                     #
############################################################################
#
#
#
################################ VERSION ###################################
from packaging.version import Version
from pyqtgraph import ErrorBarItem

VERSION = "0.3.6"                                                          #
############################################################################
#
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSpinBox, QFileDialog, QMessageBox, QGroupBox, QInputDialog, QSlider,
                             QGridLayout)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QColor, QBrush
import pyqtgraph as pg
from viewcube.utils import lsfiles, ckfiles, LoadFits, image_max_pixel
from viewcube.utils import save_spec, convert2iraf_spec, get_min_max
import astropy.io.fits as pyfits
import viewcube.version as version
from astropy import units as u
from astropy.wcs import WCS
import numpy as np
import string
import random
import sys
import os

try:
    from pyraf.iraf import splot
    PYRAF = True
except:
    PYRAF = False

try:
    import pyspeckit
    PYSPEC = True
except:
    PYSPEC = False

# Last pylab modules to import (after pyraf)
from viewcube.rgbmpl import rnorm, IntColorMap

# Configuración global de pyqtgraph
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

def get_wavelength_coordinates(w, Nwave):
    """
    Calculate the wavelength coordinates based on the input parameters.

    :param w: The input object for wavelength calculation.
    :param Nwave: The number of wavelength coordinates.
    :returns: The calculated wavelength coordinates.
    """
    w = w.sub([3])
    pix_coords = np.arange(Nwave)
    wave_coords = w.wcs_pix2world(pix_coords[:, np.newaxis], 0)
    if w.wcs.cunit[0] == "m":
        wave_coords *= 1e10
    return np.squeeze(wave_coords)

def GetIdFilter(list, filter, dfil="."):
    """
    Get the index of the first item in the list that matches the filter.

    :param list: The list of items to search.
    :param filter: The filter to apply to the list.
    :param dfil: The default filter value.
    :return: The index of the matching item in the list.
    """
    if filter is not None:
        lfil = lsfiles("*" + filter + "*", dfil)
    else:
        lfil = []
    if len(lfil) == 0:
        if filter is not None:
            print('"' + filter + '" NOT found. Set to "' + list[0] + '"')
        else:
            print('Set filter to "' + list[0] + '"')
        return 0
    else:
        print("Filter: " + ".".join(lfil[0].split(".")[:-1]))
        return list.index(lfil[0])


def GetSpaxelLimits(x, y, radius):
    """
    Calculate the mosaic limits based on the spaxel coordinates and radius.

    :param x: X coordinates of the spaxels.
    :param y: Y coordinates of the spaxels.
    :param radius: Radius multiplier for spaxel calculations.
    :return: A list containing the calculated mosaic limits [xmin, xmax, ymin, ymax].
    """
    spax_fac = radius * 7
    xbar = abs(max(x) - min(x)) * 0.5
    ybar = abs(max(y) - min(y)) * 0.5
    xmed = xbar + min(x)
    ymed = ybar + min(y)

    xfbar = 1.2 if xbar > spax_fac else 4.0
    yfbar = 1.2 if ybar > spax_fac else 4.0

    # PPAK special cases
    if len(x) == 331 or len(x) == 993:
        yfbar = 1.3
    if len(x) == 382:
        xfbar = 1.3

    xmax_mosaic = round(xmed + xbar * xfbar)
    xmin_mosaic = round(xmed - xbar * xfbar)
    ymax_mosaic = round(ymed + ybar * yfbar)
    ymin_mosaic = round(ymed - ybar * yfbar)

    return [xmin_mosaic, xmax_mosaic, ymin_mosaic, ymax_mosaic]


def GetLambdaLimits(wl, pt=0.05, wlim=None):
    """
    Calculate the wavelength limits based on the input wavelengths. Used to stablish the limits of the plot.
    Also, this function is used to set the X-axis (wavelength) limits in the spectrum plot.

    :param wl: List of wavelengths or list of wavelength ranges.
    :param pt: Percentage of wavelength range to add as padding.
    :param wlim: User-defined wavelength limits as a tuple (wlim_min, wlim_max).
    :return: A tuple containing the calculated wavelength limits (wmin - range * pt, wmax + range * pt).
    """
    if isinstance(wl, (tuple, list)):
        wl = [(np.min(item), np.max(item)) for item in wl if item is not None]
    wmin = np.min(wl)
    wmax = np.max(wl)
    if wlim is not None:
        if type(wlim) not in [list, tuple] or len(wlim) != 2:
            print("Wavelength limits should be a tuple or list of two items: ex. --> (None, 6200)")
        else:
            wlimmin, wlimmax = wlim
            if wlimmin is not None:
                wmin = wlimmin
            if wlimmax is not None:
                wmax = wlimmax
    range = abs(wmax - wmin)

    if wmin is None or wmax is None:
        raise ValueError("Wavelength limits could not be determined (wmin or wmax is None).")
    return wmin - range * pt, wmax + range * pt


def GetFluxLimits(flim, data=None):
    """
    Get the minimum and maximum flux limits based on the input. Used to stablish the limits of the plot.
    Also, this function is used to set the Y-axis (flow) limits in the spectrum plot.

    Args:
        flim: Tuple or list of two items representing the flux limits (fmin, fmax).
              If None, limits will be calculated from data.
        data: Optional numpy array to calculate limits from if flim is None.

    Returns:
        Tuple containing the minimum and maximum flux limits.

    Raises:
        ValueError: If flux limits cannot be determined.
    """
    # Initialize default values
    fmin, fmax = None, None

    # If flim is provided, use it
    if flim is not None:
        if not isinstance(flim, (list, tuple)) or len(flim) != 2:
            print("Flux limits should be a tuple or list of two items: ex. --> (None, 1e-18)")
        else:
            fmin, fmax = flim
            # Convert any string 'None' to actual None
            fmin = None if fmin == 'None' or (isinstance(fmin, str) and fmin.strip() == '') else fmin
            fmax = None if fmax == 'None' or (isinstance(fmax, str) and fmax.strip() == '') else fmax
            # Convert to float if not None
            try:
                fmin = float(fmin) if fmin is not None else None
                fmax = float(fmax) if fmax is not None else None
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid flux limit values: {e}")
                fmin, fmax = None, None

    # If limits are still None and data is provided, calculate from data
    if (fmin is None or fmax is None) and data is not None:
        try:
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                data_min = np.nanmin(valid_data)
                data_max = np.nanmax(valid_data)
                data_range = data_max - data_min

                # Set reasonable defaults if calculated values are invalid
                if np.isfinite(data_min) and np.isfinite(data_max):
                    if fmin is None:
                        fmin = max(data_min - 0.1 * data_range, 0) if data_range > 0 else 0.9 * data_min
                    if fmax is None:
                        fmax = data_max + 0.1 * data_range if data_range > 0 else 1.1 * data_max
        except Exception as e:
            print(f"Warning: Error calculating flux limits from data: {e}")

    # Final validation
    if fmin is None or fmax is None:
        # If we still don't have valid limits, provide some defaults
        if fmin is None:
            fmin = 0.0
        if fmax is None:
            fmax = 1.0
        print(f"Warning: Using default flux limits: ({fmin}, {fmax})")

    # Ensure fmin < fmax
    if fmin >= fmax:
        fmin, fmax = fmax, fmin

    return float(fmin), float(fmax)


def PRectangle(x, y, r):
    """
    Calculate the coordinates of a rectangle based on the input parameters.

    :param x: X-coordinate or list of X-coordinates of the rectangle.
    :param y: Y-coordinate or list of Y-coordinates of the rectangle.
    :param r: The radius of the rectangle.
    :returns: Two arrays representing the X and Y coordinates of the rectangle vertices.
    """
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        y = np.array(y)
    if isinstance(x, (int, float)):
        xv = x + np.array([0.0, 0.0, r, r])
        yv = y + np.array([0.0, r, r, 0.0])
    else:
        xv = x[:, np.newaxis] + np.array([0.0, 0.0, r, r])
        yv = y[:, np.newaxis] + np.array([0.0, r, r, 0.0])
    return xv, yv


def tmpName(prefix="tmp", char=8, suffix="fits"):
    """
    Generate a temporary filename.

    :param prefix: The prefix for the filename (default is 'tmp').
    :param char: The number of characters for the random string (default is 8).
    :param suffix: The suffix for the filename (default is 'fits').
    :return: A string representing the temporary filename in the format 'prefix_randomString.suffix'.
    """
    schar = "".join(random.choice(string.ascii_letters + string.digits) for i in range(char))
    return "%s_%s.%s" % (prefix, schar, suffix)

class CubeViewer(QWidget):
    def eventFilter(self, obj, event):
        """
        Filter events for the widget.
        
        Args:
            obj: The object that received the event
            event: The event
            
        Returns:
            bool: True if the event was handled, False otherwise
        """
        # Handle key press events
        if event.type() == QEvent.KeyPress:
            self.keyPressEvent(event)
            return True
            
        # Handle mouse press events
        elif event.type() == QEvent.MouseButtonPress:
            self.mousePressEvent(event)
            return True
            
        # Handle mouse move events
        elif event.type() == QEvent.MouseMove:
            self.mouseMoveEvent(event)
            return True
            
        # Handle mouse release events
        elif event.type() == QEvent.MouseButtonRelease:
            self.mouseReleaseEvent(event)
            return True
            
        return super().eventFilter(obj, event)

    def __init__(self, name_fits, **kwargs):
        try:
            super().__init__()
            
            # Inicializar variables básicas
            self.kwargs = kwargs
            self.name_fits = name_fits
            self.fits_data = None
            self.data = None
            self.nx = 0
            self.ny = 0
            self.wl = None
            self.spec = None
            self.speccom = None
            
            # Configurar widgets primero
            self.setup_ui()
            
            # Cargar datos si se proporciona un archivo
            if self.name_fits:
                self.load_data()
                
        except Exception as e:
            QMessageBox.critical(self, "Error de Inicialización", 
                               f"Error al inicializar el visor: {str(e)}")
            raise

    def setup_ui(self):
        """Configura la interfaz de usuario"""
        try:
            # Crear layout principal
            self.main_layout = QVBoxLayout()
            self.setLayout(self.main_layout)
            
            # Configurar widgets
            self.setup_spaxel_widget()
            self.setup_spectral_widget()
            
            # Añadir widgets al layout
            self.main_layout.addWidget(self.spaxel_widget)
            self.main_layout.addWidget(self.spectrum_widget)
            
            # Configurar eventos
            self.movement_and_mouse_events()
            self.keyboard_events()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error configurando la interfaz: {str(e)}")
            raise

    def load_data(self):
        """Carga los datos del archivo FITS"""
        if not self.name_fits:
            QMessageBox.warning(self, "Advertencia", "No se ha especificado ningún archivo FITS")
            return
            
        try:
            # Verificar que el archivo existe
            if not os.path.exists(self.name_fits):
                raise FileNotFoundError(f"No se encontró el archivo: {self.name_fits}")
                
            # Cargar archivo FITS usando LoadFits
            self.fits_data = LoadFits(
                self.name_fits,
                exdata=self.kwargs.get('exdata'),
                exhdr=self.kwargs.get('exhdr', 0),
                exerror=self.kwargs.get('exerror'),
                exflag=self.kwargs.get('exflag'),
                ivar=self.kwargs.get('ivar', False)
            )
            
            if self.fits_data is None:
                raise ValueError("No se pudo cargar el archivo FITS")
                
            if not hasattr(self.fits_data, 'data') or self.fits_data.data is None:
                raise ValueError("El archivo FITS no contiene datos válidos")
                
            # Inicializar dimensiones
            self.data = self.fits_data.data
            if len(self.data.shape) != 3:
                raise ValueError("Los datos deben ser un cubo 3D")
                
            self.nx = self.data.shape[2]
            self.ny = self.data.shape[1]
            
            # Inicializar wavelength si existe
            if hasattr(self.fits_data, 'wave'):
                self.wl = self.fits_data.wave
            else:
                self.wl = np.arange(self.data.shape[0])
            
            # Inicializar espectro promedio
            self.spec = np.nanmean(self.data, axis=(1,2))
            
            # Inicializar datos de comparación si existen
            if self.kwargs.get('fitscom'):
                try:
                    comp_data = LoadFits(
                        self.kwargs['fitscom'],
                        exdata=self.kwargs.get('exdata'),
                        exhdr=self.kwargs.get('exhdr', 0)
                    )
                    if comp_data is not None and comp_data.data is not None:
                        self.speccom = np.nanmean(comp_data.data, axis=(1,2))
                    else:
                        self.speccom = None
                except Exception as e:
                    print(f"Advertencia: No se pudieron cargar los datos de comparación: {e}")
                    self.speccom = None
            else:
                self.speccom = None
            
            # Actualizar visualizaciones
            self.update_visualizations()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error cargando archivo FITS: {str(e)}")
            self.fits_data = None
            self.data = None
            self.nx = 0
            self.ny = 0
            self.wl = None
            self.spec = None
            self.speccom = None

    def setup_spaxel_widget(self):
        """Configura el widget para visualización de spaxels"""
        try:
            self.spaxel_widget = pg.PlotWidget()
            self.spaxel_widget.setAspectLocked(True)
            self.spaxel_widget.showGrid(x=True, y=True)
            self.spaxel_widget.setMouseEnabled(x=True, y=True)
            self.spaxel_widget.enableAutoRange()
            self.spaxel_widget.setMenuEnabled(True)
            self.spaxel_widget.setDownsampling(auto=True, mode='peak')
            self.spaxel_widget.setLabel('left', 'Y')
            self.spaxel_widget.setLabel('bottom', 'X')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error configurando widget de spaxels: {str(e)}")
            raise

    def setup_spectral_widget(self):
        """Configura el widget para visualización de espectros"""
        try:
            self.spectrum_widget = pg.PlotWidget()
            self.spectrum_widget.showGrid(x=True, y=True)
            self.spectrum_widget.setMouseEnabled(x=True, y=True)
            self.spectrum_widget.enableAutoRange()
            self.spectrum_widget.setMenuEnabled(True)
            self.spectrum_widget.setLabel('left', 'Flux')
            self.spectrum_widget.setLabel('bottom', 'Wavelength')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error configurando widget de espectros: {str(e)}")
            raise

    def update_visualizations(self):
        """Actualiza las visualizaciones de spaxel y espectro"""
        try:
            if not hasattr(self, 'fits_data') or self.fits_data is None:
                return
            
            self.update_spaxel_view()
            self.update_spectrum_view()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error actualizando visualizaciones: {str(e)}")
        
    def update_spaxel_view(self):
        """Actualiza la vista de spaxels"""
        if hasattr(self, 'fits_data') and self.fits_data is not None:
            try:
                # Limpiar vista anterior
                self.spaxel_widget.clear()
                
                # Crear imagen
                img = pg.ImageItem(self.fits_data.data[0])
                self.spaxel_widget.addItem(img)
                
                # Ajustar límites
                self.spaxel_widget.autoRange()
                
                # Agregar colorbar si está habilitado
                try:
                    if getattr(self, 'colorbar', True):  # Si no existe, asume True
                        colorbar = pg.ColorBarItem(
                            values=(np.nanmin(self.fits_data.data[0]), np.nanmax(self.fits_data.data[0])),
                            colorMap='viridis'
                        )
                        colorbar.setImageItem(img)
                except Exception as e:
                    print(f"Error al crear colorbar: {e}")
            except Exception as e:
                print(f"Error actualizando vista de spaxels: {e}")
                
    def update_spectrum_view(self):
        """Actualiza la vista del espectro"""
        if hasattr(self, 'fits_data') and self.fits_data.data is not None:
            # Limpiar vista anterior
            self.spectrum_widget.clear()
            
            # Si hay wavelength, usarlo, si no crear array
            if hasattr(self.fits_data, 'wave') and self.fits_data.wave is not None:
                x = self.fits_data.wave
            else:
                x = np.arange(self.fits_data.data.shape[0])
                
            # Obtener espectro promedio
            y = np.mean(self.fits_data.data, axis=(1,2))
            
            # Aplicar límites si están definidos
            if self.wlim is not None:
                wmin, wmax = GetLambdaLimits(x, wlim=self.wlim)
                mask = (x >= wmin) & (x <= wmax)
                x = x[mask]
                y = y[mask]
                
            if self.flim is not None:
                fmin, fmax = GetFluxLimits(self.flim)
                if fmin is not None:
                    y = np.maximum(y, fmin)
                if fmax is not None:
                    y = np.minimum(y, fmax)
            
            # Plotear espectro
            self.spectrum_widget.plot(x, y, pen=pg.mkPen('b', width=2))
            
    def on_spaxel_click(self, event):
        """Maneja el evento de clic en la vista de spaxels"""
        pos = event.scenePos()
        view_pos = self.spaxel_widget.getPlotItem().vb.mapSceneToView(pos)
        x, y = int(view_pos.x()), int(view_pos.y())
        
        if hasattr(self, 'fits_data') and self.fits_data.data is not None:
            if 0 <= x < self.fits_data.data.shape[2] and 0 <= y < self.fits_data.data.shape[1]:
                # Actualizar espectro seleccionado
                self.plot_selected_spectrum(x, y)
                
    def plot_selected_spectrum(self, x, y):
        """Plotea el espectro del spaxel seleccionado"""
        if hasattr(self, 'fits_data') and self.fits_data.data is not None:
            # Obtener espectro del spaxel
            spectrum = self.fits_data.data[:, y, x]
            
            # Obtener wavelength si existe
            if hasattr(self.fits_data, 'wave') and self.fits_data.wave is not None:
                wavelength = self.fits_data.wave
            else:
                wavelength = np.arange(len(spectrum))
                
            # Actualizar vista del espectro
            self.spectrum_widget.clear()
            self.spectrum_widget.plot(wavelength, spectrum, pen=pg.mkPen('r', width=2))
            
    def get_info(self):
        """Retorna información sobre el cubo cargado"""
        if hasattr(self, 'fits_data'):
            info = []
            if hasattr(self.fits_data, 'data'):
                info.append(f"Data shape: {self.fits_data.data.shape}")
            if hasattr(self.fits_data, 'wave'):
                info.append(f"Wavelength range: {self.fits_data.wave[0]:.2f} - {self.fits_data.wave[-1]:.2f}")
            return "\n".join(info)
        return "No data loaded"
        
    def PlotSpec(self):
        """Muestra el espectro actual"""
        if hasattr(self, 'fits_data') and self.fits_data.data is not None:
            # Crear una nueva ventana para el espectro
            spectrum_window = QWidget()
            layout = QVBoxLayout(spectrum_window)
            
            # Crear widget de gráfico
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            plot_widget.showGrid(x=True, y=True)
            plot_widget.setLabel('left', 'Flux')
            plot_widget.setLabel('bottom', 'Wavelength')
            
            # Obtener datos
            if hasattr(self.fits_data, 'wave') and self.fits_data.wave is not None:
                x = self.fits_data.wave
            else:
                x = np.arange(self.fits_data.data.shape[0])
                
            y = np.mean(self.fits_data.data, axis=(1,2))
            
            # Plotear datos
            plot_widget.plot(x, y, pen=pg.mkPen('b', width=2))
            
            # Agregar widget al layout
            layout.addWidget(plot_widget)
            
            # Mostrar ventana
            spectrum_window.setWindowTitle("Spectrum Viewer")
            spectrum_window.resize(800, 600)
            spectrum_window.show()
            
    def updateAx1(self, color=True):
        """Actualiza la vista principal de spaxels"""
        self.update_spaxel_view()
        
    def plotResidualMap(self):
        """Muestra el mapa de residuos"""
        if hasattr(self, 'fits_data') and self.fits_data.data is not None and self.fitscom is not None:
            try:
                # Cargar datos de comparación
                comp_data = LoadFits(self.fitscom).data
                
                # Calcular residuos
                residuals = self.fits_data.data - comp_data
                
                # Crear nueva ventana
                residual_window = QWidget()
                layout = QVBoxLayout(residual_window)
                
                # Crear widget de gráfico
                plot_widget = pg.PlotWidget()
                plot_widget.setBackground('w')
                
                # Crear imagen de residuos
                img = pg.ImageItem(residuals[0])
                plot_widget.addItem(img)
                
                # Agregar colorbar
                colorbar = pg.ColorBarItem(
                    values=(np.min(residuals[0]), np.max(residuals[0])),
                    colorMap='viridis'
                )
                colorbar.setImageItem(img)
                
                # Agregar widget al layout
                layout.addWidget(plot_widget)
                
                # Mostrar ventana
                residual_window.setWindowTitle("Residual Map")
                residual_window.resize(800, 600)
                residual_window.show()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error calculando residuos: {str(e)}")
                
    def ResidualViewer(self, event=None):
        """Muestra el visor de residuos"""
        self.plotResidualMap()
        
    def WindowManager(self):
        """Muestra el administrador de ventanas"""
        # Crear ventana
        manager_window = QWidget()
        layout = QVBoxLayout(manager_window)
        
        # Crear controles
        # Límites de longitud de onda
        wave_group = QGroupBox("Wavelength Limits")
        wave_layout = QHBoxLayout(wave_group)
        
        wave_min = QSpinBox()
        wave_min.setRange(0, 10000)
        wave_min.setValue(4000)
        
        wave_max = QSpinBox()
        wave_max.setRange(0, 10000)
        wave_max.setValue(7000)
        
        wave_layout.addWidget(QLabel("Min:"))
        wave_layout.addWidget(wave_min)
        wave_layout.addWidget(QLabel("Max:"))
        wave_layout.addWidget(wave_max)
        
        # Botón de aplicar
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(lambda: self.apply_wave_limits(wave_min.value(), wave_max.value()))
        
        # Agregar widgets al layout
        layout.addWidget(wave_group)
        layout.addWidget(apply_button)
        
        # Mostrar ventana
        manager_window.setWindowTitle("Window Manager")
        manager_window.resize(400, 200)
        manager_window.show()
        
    def apply_wave_limits(self, wmin, wmax):
        """Aplica los límites de longitud de onda"""
        self.wlim = [wmin, wmax]
        self.update_spectrum_view()
        
    def SaveFile(self):
        """Guarda el espectro actual"""
        if hasattr(self, 'fits_data') and self.fits_data.data is not None:
            try:
                # Obtener nombre de archivo
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Guardar espectro",
                    "",
                    "Text Files (*.txt);;FITS Files (*.fits)"
                )
                
                if file_path:
                    # Obtener datos
                    if hasattr(self.fits_data, 'wave') and self.fits_data.wave is not None:
                        x = self.fits_data.wave
                    else:
                        x = np.arange(self.fits_data.data.shape[0])
                        
                    y = np.mean(self.fits_data.data, axis=(1,2))
                    
                    # Guardar archivo
                    if file_path.endswith('.txt'):
                        np.savetxt(file_path, np.column_stack((x, y)))
                    else:
                        hdu = pyfits.PrimaryHDU(np.column_stack((x, y)))
                        hdu.writeto(file_path, overwrite=True)
                        
                    QMessageBox.information(self, "Éxito", "Archivo guardado correctamente")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error guardando archivo: {str(e)}")
                
    def Sonification(self):
        """Muestra la interfaz de sonificación"""
        QMessageBox.information(self, "Info", "Sonification feature not implemented in PyQt5 version yet")

    def getSpec(self, x, y):
        """Obtiene el espectro para un spaxel específico"""
        if hasattr(self, 'data') and self.data is not None:
            if len(self.data.shape) == 3:
                return self.data[:, y, x]
            else:
                return self.data[y, x]
        return None

    def res2pix(self, x, y):
        return int((x / self.sr) + self.x_ref), int((y / self.sr) + self.y_ref)

    def pix2res(self, x, y):
        return (x - self.x_ref) * self.sr, (y - self.y_ref) * self.sr

    def vredshift(self, cz):
        return self.wl / (1.0 + (cz / self.c))

    def get_ref_pix(self, mode="max", **kwargs):
        if mode.lower() == "max":
            image = np.nanmedian(self.dat, axis=0)
            self.y_ref, self.x_ref = image_max_pixel(image, **kwargs)
        else:
            self.y_ref = self.crpix2
            self.x_ref = self.crpix1
        self.ext = [
            -self.x_ref * self.sr,
            (self.xx - self.x_ref) * self.sr,
            -self.y_ref * self.sr,
            (self.yy - self.y_ref) * self.sr,
        ]

    def onselect(self):
        # Obtener los límites del ROI en coordenadas de plot
        rect = self.rect_roi.parentBounds()  # QRectF
        x0, y0 = rect.left(), rect.top()
        x1, y1 = rect.right(), rect.bottom()
        # Ordena para asegurar que x0 < x1, y0 < y1
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        # posibilidad de transformación a índices de array
        ix0, ix1 = int(round(x0)), int(round(x1))
        iy0, iy1 = int(round(y0)), int(round(y1))

        # Itera sobre los píxeles/spaxels seleccionados
        selected = []
        for x in range(ix0, ix1):
            for y in range(iy0, iy1):
                selected.append((x, y))
        self.selected_spaxels = selected

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Asterisk:
            # Limpiar y actualizar los widgets de spaxel y espectro
            self.spaxel_widget.clear()
            self.spectrum_widget.clear()

            # Mantener rango de color actual
            if hasattr(self, 'color') and self.color is not None:
                cmin, cmax = np.nanmin(self.color), np.nanmax(self.color)
            else:
                cmin, cmax = 0, 1
            # Volver a mostrar la imagen y título
            img = pg.ImageItem()
            img.setImage(self.color.T, levels=(cmin, cmax), opacity=self.palpha)
            if hasattr(self, 'ext'):
                img.setRect(
                    pg.QtCore.QRectF(self.ext[0], self.ext[2], self.ext[1] - self.ext[0], self.ext[3] - self.ext[2]))
                self.spaxel_widget.setXRange(self.ext[0], self.ext[1])
                self.spaxel_widget.setYRange(self.ext[2], self.ext[3])
            self.spaxel_widget.addItem(img)
            title = pg.TextItem(self.bname_fits, anchor=(0.5, 1.0), color='k')
            self.spaxel_widget.addItem(title)
            title.setPos((self.ext[0] + self.ext[1]) / 2, self.ext[3])
            # Restaurar colorbar si corresponde
            if getattr(self, 'colorbar', False):
                hist = pg.HistogramLUTItem()
                hist.setImageItem(img)
                hist.gradient.loadPreset('viridis')
                self.spaxel_widget.scene().addItem(hist)
            # Restaurar modo DS9-like si corresponde
            if getattr(self, 'iclm', False):
                # Aquí deberías restaurar el estado de tu IntColorMap si lo tienes implementado
                # Por ejemplo, podrías guardar el rango anterior y restaurarlo
                pass
            self.list = []
            self.pbline = None
            # Actualizar espectro (puedes llamar a tu método de actualización)
            self.update_spectrum_view()
            return

        # Cambiar modo (equivalente a "s")
        if key == Qt.Key_S:
            self.mode = not self.mode
            return

        # Sonificación (equivalente a "h")
        if key == Qt.Key_H:
            if not self.soni_start:
                self.soni_start = True
                self.Sonification()
            self.soni_mode = not self.soni_mode
            if self.sc is None:
                self.soni_mode = False
            if not self.soni_mode and self.sc is not None and getattr(self.sc, 'cs', None) is not None:
                self.sc.stop_sound()
            return

        # Guardar espectros seleccionados (equivalente a "S")
        if key == Qt.Key_S and hasattr(self, 'list') and len(self.list) > 0:
            self.SaveFile()
            return

        # Window Manager (equivalente a "w")
        if key == Qt.Key_W:
            self.WindowManager()
            return

        # Límites de lambda (equivalente a "l")
        if key == Qt.Key_L:
            self.LambdaLimits()
            return

        # Límites de flujo (equivalente a "Y")
        if key == Qt.Key_Y:
            self.FluxLimits()
            return

        # Mostrar puntos individuales (equivalente a "I")
        if key == Qt.Key_I and hasattr(self.fobj, 'K') and self.fobj.K is not None:
            self.view_pintspec = not self.view_pintspec
            return

        # Salir (equivalente a "q")
        if key == Qt.Key_Q:
            if self.sc is not None and getattr(self.sc, 'cs', None) is not None:
                self.sc.close_sound()
            sys.exit()

    def setWindowTitle(self, widget, label):
        try:
            widget.setWindowTitle(label)
        except Exception:
            pass

    def SaveFile(self):
        """
        Guarda los espectros seleccionados (integrado e individuales) usando diálogos de PyQt5.
        """
        # Mostrar opciones actuales en un diálogo informativo
        msg = (
            f'***** Actual Save File Options for "{self.name_fits}" *****\n'
            f"fits = {self.fits} | txt = {self.txt}\n"
            f"integrated = {self.integrated} | individual = {self.individual}\n"
            "Se guardarán los espectros seleccionados según las opciones actuales."
        )
        QMessageBox.information(self, "Opciones de guardado", msg)

        # Diálogo para elegir nombre base del archivo
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar espectros",
            self.root,
            "FITS (*.fits);;Texto (*.txt);;Todos los archivos (*)"
        )

        if not fname:
            QMessageBox.information(self, "Guardar espectros", "Nada que guardar.")
            return

        # Comprobar opciones
        if (not self.txt and not self.fits) or (not self.integrated and not self.individual):
            QMessageBox.warning(self, "Guardar espectros", "Nada que guardar: revisa las opciones de guardado.")
            return

        # Sufijo para espectro integrado
        stint = "_int" if self.root == fname else ""

        # Guardar espectro integrado
        if self.integrated:
            infotxt = [
                f'Integrated Spectra extracted from: "{self.name_fits}"',
                "Sum of Spaxels (ID): " + " | ".join([str(id) for id in self.list]),
            ]
            infohd = [
                ["3DVWR_1", infotxt[0]],
                ["3DVWR_2", infotxt[1]],
                ["CRVAL1", self.crval],
                ["CDELT1", self.cdelt],
            ]
            # Sumar espectros seleccionados
            self.intspec = np.array([self.dat[:, y, x] for x, y in self.list]).sum(0)
            # Guardar archivo
            save_spec(
                self.wl,
                self.intspec,
                fname + stint,
                fits=self.fits,
                txt=self.txt,
                hd=self.hd,
                infohd=infohd,
                infotxt=infotxt,
            )

        # Guardar espectros individuales
        if self.individual:
            for item in self.list:
                infotxt = [
                    f'Spectra extracted from: "{self.name_fits}"',
                    f"Spaxel (ID): {item}",
                ]
                infohd = [
                    ["3DVWR_1", infotxt[0]],
                    ["3DVWR_2", infotxt[1]],
                    ["CRVAL1", self.crval],
                    ["CDELT1", self.cdelt],
                ]
                x, y = item
                strl = (f"%0{self.fx}i_%0{self.fy}i") % (x, y)
                save_spec(
                    self.wl,
                    self.dat[:, y, x],
                    f"{fname}_{strl}",
                    fits=self.fits,
                    hd=self.hd,
                    txt=self.txt,
                    infohd=infohd,
                    infotxt=infotxt,
                )

        QMessageBox.information(self, "Guardar espectros", "Archivos guardados correctamente.")

    def LambdaLimits(self):
        """Establece los límites de longitud de onda en el espectro usando PyQt5"""
        # Diálogo para entrada de límites
        text, ok = QInputDialog.getText(
            self,
            "Límites de longitud de onda",
            "Ingrese límites (ej: 4000, 7000)\nNone para automático:",
            text=f"{self.awlmin}, {self.awlmax}" if hasattr(self, 'awlmin') else ""
        )

        if ok:
            try:
                # Parsear entrada
                parts = [p.strip() for p in text.split(',')]
                wlim = []
                for p in parts[:2]:  # Tomar máximo 2 valores
                    if p.lower() in ('none', ''):
                        wlim.append(None)
                    else:
                        wlim.append(float(p))

                # Llamar a la función original de cálculo
                self.awlmin, self.awlmax = GetLambdaLimits(
                    (self.wl, self.wl2),
                    0.05,
                    wlim=tuple(wlim) if len(wlim) > 0 else None
                )
                if self.awlmin is None or self.awlmax is None:
                    raise ValueError("Límites de lambda no válidos")
                self.spectrum_widget.setXRange(self.awlmin, self.awlmax)

                # Actualizar vista en PyQtGraph
                self.spectrum_widget.setXRange(self.awlmin, self.awlmax)

                # Mensaje de confirmación
                QMessageBox.information(
                    self,
                    "Límites actualizados",
                    f"Límites lambda: ({self.awlmin:.1f}, {self.awlmax:.1f})"
                )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Entrada inválida: {str(e)}\nEjemplo válido: '4000, 7000'"
                )

    def FluxLimits(self):
        """Establece los límites de flujo en el espectro con validación mejorada."""
        # Get current values or reasonable defaults
        current_min = f"{self.fmin:.2e}" if hasattr(self, 'fmin') and self.fmin is not None else ""
        current_max = f"{self.fmax:.2e}" if hasattr(self, 'fmax') and self.fmax is not None else ""

        # Show input dialog with current values
        text, ok = QInputDialog.getText(
            self,
            "Límites de flujo",
            "Ingrese límites de flujo (mín, máx):\n"
            "Ejemplos:\n"
            "- '1e-16, 1e-15' para límites fijos\n"
            '- "None, 1e-15" para límite superior fijo\n'
            '- "1e-16, None" para límite inferior fijo\n'
            '- "auto" para ajuste automático',
            text=f"{current_min}, {current_max}"
        )

        if not ok:
            return

        try:
            # Handle auto mode
            if text.strip().lower() == 'auto':
                self.fmin, self.fmax = None, None
                if hasattr(self, 'spec') and self.spec is not None:
                    self.fmin, self.fmax = GetFluxLimits(None, data=self.spec)
            else:
                # Parse user input
                parts = [p.strip() for p in text.split(',')]
                if len(parts) != 2:
                    raise ValueError("Debe ingresar exactamente dos valores separados por coma")

                # Parse min and max values
                flim = []
                for p in parts:
                    p = p.strip()
                    if p.lower() in ('none', ''):
                        flim.append(None)
                    else:
                        try:
                            flim.append(float(p))
                        except ValueError:
                            raise ValueError(f"Valor no válido: {p}")

                # Get new limits
                self.fmin, self.fmax = GetFluxLimits(flim, getattr(self, 'spec', None))

            # Update plot
            if hasattr(self, 'spectrum_widget') and self.spectrum_widget is not None:
                if self.fmin is not None and self.fmax is not None:
                    self.spectrum_widget.setYRange(self.fmin, self.fmax)

            # Show success message
            QMessageBox.information(
                self,
                "Límites actualizados",
                f"Límites de flujo actualizados:\n"
                f"Mínimo: {self.fmin:.2e}\n"
                f"Máximo: {self.fmax:.2e}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error al actualizar los límites de flujo:\n{str(e)}\n\n"
                "Formato esperado: 'min, max' o 'auto'\n"
                "Ejemplo: '1e-16, 1e-15' o 'None, 1e-15'"
            )

    def Redshift(self):
        """
        Solicita el redshift al usuario, actualiza la longitud de onda y refresca la visualización.
        """
        # Diálogo para entrada de redshift en km/s
        cz, ok = QInputDialog.getText(
            self,
            "Redshift",
            "Introduce el redshift en km/s (Enter para abortar):"
        )
        if not ok or len(cz.strip()) == 0:
            QMessageBox.information(self, "Redshift", "No se ha introducido ningún valor de redshift.")
            return

        try:
            velocity = float(cz)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Valor inválido: {cz}\n{str(e)}")
            return

        # Actualizar la longitud de onda desplazada
        self.wl = self.vredshift(velocity)
        if self.velocity is None:
            self.velocity = velocity
        if self.redshift is None:
            self.redshift = velocity / self.c
        if self.orig_wl_rest is None:
            self.orig_wl_rest = self.wl

        # Si hay un spaxel seleccionado, actualizar la visualización
        if self.ix is not None and self.iy is not None:
            self.set_fff(verb=False, dl=self.gdl)
            self.PlotSpec()
            self.updatePatch()
            # Actualizar widgets de PyQtGraph
            self.update_spaxel_view()
            self.update_spectrum_view()

    def RestWave(self):
        """
        Alterna entre longitud de onda rest-frame y observada,
        actualiza los datos y refresca la visualización en PyQtGraph.
        """
        if self.orig_wl_rest is not None:
            if self.wrest and self.velocity is not None:
                msg = (
                    f"Rest Wavelength (redshift = {self.redshift:8.5f} | "
                    f"velocity = {self.velocity:7.1f} km/s)"
                )
                self.wl = self.orig_wl_rest
            elif not self.wrest and self.velocity is not None:
                msg = (
                    f"Observed Wavelength (redshift = {self.redshift:8.5f} | "
                    f"velocity = {self.velocity:7.1f} km/s)"
                )
                self.wl = self.orig_wl
            else:
                msg = "No velocity information available."
                QMessageBox.information(self, "RestWave", msg)
                return

            # Mensaje informativo (opcional)
            QMessageBox.information(self, "RestWave", msg)

            # Actualizar espectro si hay spaxel seleccionado
            if self.ix is not None and self.iy is not None:
                self.set_fff(verb=False, dl=self.gdl)
                self.PlotSpec()
                self.updatePatch()
                self.update_spaxel_view()
                self.update_spectrum_view()

            # Alternar wrest
            self.wrest = not self.wrest

    def SynthSpec(self):
        """Alterna la visualización de espectros sintéticos en PyQtGraph"""
        if self.syncom:
            self.syncom = False
            if self.syn is None:
                QMessageBox.warning(self, "Advertencia", "No se encontraron datos de espectros sintéticos")
                return
            # Restaurar datos originales
            self.fitscom = None
            if self.ix is not None and self.iy is not None:
                self.spectrum_widget.clear()
                self.plot_selected_spectrum(self.ix, self.iy)
                self.updatePassBand()
        else:
            self.syncom = True
            if self.syn is not None:
                self.fitscom = "Espectros Sintéticos"
                self.wl2 = self.wl
                self.dat2 = self.syn
                if self.ix is not None and self.iy is not None:
                    # Limpiar y volver a plotear
                    self.spectrum_widget.clear()
                    # Espectro original
                    self.spectrum_widget.plot(
                        self.wl,
                        self.dat[:, self.iy, self.ix],
                        pen=pg.mkPen(self.cspec, width=self.lspec)
                    )
                    # Espectro sintético
                    self.spectrum_widget.plot(
                        self.wl2,
                        self.dat2[:, self.iy, self.ix],
                        pen=pg.mkPen(self.ccom, width=self.lcom),
                        name=self.fitscom
                    )
                    self.updatePassBand()
            else:
                QMessageBox.warning(self, "Advertencia", "No se encontraron datos de espectros sintéticos")
                self.syncom = False

    def ErrorSpec(self):
        """Alterna la visualización de barras de error en el espectro"""
        if self.ix is None or self.iy is None:
            return

        # Alternar estado de error
        self.errcom = not self.errcom

        # Verificar si hay datos de error
        if self.errcom and self.err is None:
            QMessageBox.warning(self, "Error", "No se encontraron datos de error")
            self.errcom = False
            return

        # Eliminar barras de error anteriores si existen
        if hasattr(self, 'error_bars'):
            for item in self.error_bars:
                self.spectrum_widget.removeItem(item)
            self.error_bars = []

        # Añadir nuevas barras de error
        if self.errcom and self.err is not None:
            # Obtener datos del espectro principal
            x = self.wl
            y = self.dat[:, self.iy, self.ix]
            y_err = self.err[:, self.iy, self.ix]

            # Crear ErrorBarItem para espectro principal
            error_bar = ErrorBarItem(
                x=np.array(x),
                y=np.array(y),
                top=np.array(y_err),
                bottom=np.array(y_err),
                beam=0.5,
                pen=pg.mkPen('grey', width=1)
            )
            self.spectrum_widget.addItem(error_bar)
            self.error_bars.append(error_bar)

            # Manejar datos de comparación si existen
            if self.fitscom is not None and self.err2 is not None:
                x2 = self.wl2
                y2 = self.dat2[:, self.iy, self.ix]
                y2_err = self.err2[:, self.iy, self.ix]

                error_bar2 = ErrorBarItem(
                    x=np.array(x2),
                    y=np.array(y2),
                    top=np.array(y2_err),
                    bottom=np.array(y2_err),
                    beam=0.5,
                    pen=pg.mkPen('grey', width=1)
                )
                self.spectrum_widget.addItem(error_bar2)
                self.error_bars.append(error_bar2)

        # Actualizar visualización
        if not self.errcom:
            # Limpiar y volver a plotear sin errores
            self.spectrum_widget.clear()
            self.PlotSpec()
            self.updatePassBand()

    def ResSpec(self):
        """Alterna entre datos originales y residuos en PyQtGraph"""
        # Obtener estado actual de las teclas desde el eventFilter
        key = self.last_key  # Asume que guardas la última tecla en self.last_key

        if key == Qt.Key_R or key == Qt.Key_U:
            if key == Qt.Key_R:
                self.specres = not self.specres
            if key == Qt.Key_U:
                self.specfres = not self.specfres

            if self.specres and self.K is None:
                QMessageBox.warning(self, "Error", "No se encontraron datos residuales")
                self.specres = False
                return

            # Actualizar fuente de datos
            if self.specres:
                self.res = self.K.fres if self.specfres else self.K.res
                self.pres = self.res.copy()
                self.pres[~self.flag] = np.nan
                self.dcolor = self.res
            else:
                self.dcolor = self.dat

            # Actualizar visualización espectral
            self.spectrum_widget.clear()
            self.PlotSpec()
            self.updatePassBand()

            # Actualizar vista de spaxels
            self.update_spaxel_view()

    def BoxFilter(self, wini=4500.0, wend=5000.0, dl=1.0):
        lf = np.arange(wini - dl, wend + dl, dl)
        ff = np.ones(lf.shape)
        ff[0] = 0.0
        ff[-1] = 0.0
        return lf, ff

    def IntFilter(
        self,
        ifi,
        fil,
        lm,
        dat,
        pasb=False,
        verb=True,
        dfil="",
        dl=None,
        center=False,
        remove_cont=False,
    ):
        def ftrapz(lm, fff, dat, ax=None):
            if dat.ndim == 3:
                # tile(lm,80*75).reshape((1970,75,80))
                lmd = lm[:, np.newaxis, np.newaxis]
                fffd = fff[:, np.newaxis, np.newaxis]
                ax = 0 if ax is None else ax
            else:
                ax = 1 if ax is None else ax
                lmd = lm
                fffd = fff
            return np.trapz(lmd * fffd * dat, lm, axis=ax) / np.trapz(fff * lm, lm)

        # Turn Off: "Warning: overflow encountered in multiply"
        np.seterr(all="ignore")
        if fil is not None:
            lf, ff = np.loadtxt(os.path.join(dfil, fil[ifi]), unpack=True, usecols=(0, 1))
        else:
            lf, ff = self.BoxFilter()
        if verb is True and fil is not None:
            print("Selected Filter: " + fil[ifi])
        iffmax = np.argmax(ff)
        if center:
            wmax = lf[iffmax]
            wcen = (lm.min() + lm.max()) / 2.0
            lf += wcen - wmax
        if dl is not None:
            lf = lf + dl
        self.wmax = lf[iffmax]
        fff = np.interp(lm, lf, ff)
        ival = ftrapz(lm, fff, dat)
        if remove_cont:
            if verb:
                print(">>> Continuum removed")
            dlw = lf.max() - lf.min()
            lfff = np.interp(lm, lf - dlw, ff)
            rfff = np.interp(lm, lf + dlw, ff)
            cval = (ftrapz(lm, lfff, dat) + ftrapz(lm, rfff, dat)) / 2.0
            ival -= cval
        if pasb is True:
            return ival, fff
        else:
            return ival

    def set_fff(self, **kwargs):
        kwargs.pop("pasb", None)
        dpars = dict(
            ifi=self.ifil,
            fil=self.list_filters,
            pasb=True,
            dfil=self.dfilter,
            center=self.cfilter,
            remove_cont=self.remove_cont,
        )
        dpars.update(kwargs)
        self.color, self.fff = self.IntFilter(lm=self.wl, dat=self.dcolor, **dpars)
        if self.fitscom is not None and "Synthetic" not in self.fitscom:
            dpars["verb"] = False
            self.color2, self.fff2 = self.IntFilter(lm=self.wl2, dat=self.dat2, **dpars)

    def updatePatch(self):
        """
        Actualiza el mapa de color (spaxel map) en PyQtGraph tras un cambio de filtro, residual, etc.
        """
        # Selecciona el array de color adecuado según el contexto
        color = self.color
        if not (self.fitscom is None or (self.fitscom is not None and "Synthetic" in str(self.fitscom))):
            # Decide si usar color principal o secundario según el rango de wmax
            if (self.wl[0] <= self.wmax) and (self.wmax <= self.wl[-1]):
                color = self.color
            if (self.wl2[0] <= self.wmax) and (self.wmax <= self.wl2[-1]):
                color = self.color2

        # Calcula los límites de color
        self.cmin, self.cmax = get_min_max(color)

        # Actualiza el ImageItem del widget de spaxels
        # Busca el ImageItem en el widget (asumiendo que es el primero añadido)
        img_items = [item for item in self.spaxel_widget.listDataItems() if isinstance(item, pg.ImageItem)]
        if img_items:
            img = img_items[0]
            img.setImage(color.T, levels=(self.cmin, self.cmax), opacity=self.palpha)
        else:
            # Si no existe, crea uno nuevo
            img = pg.ImageItem(color.T, levels=(self.cmin, self.cmax), opacity=self.palpha)
            self.spaxel_widget.addItem(img)

    def updatePassBand(self, remove=False):
        # Eliminar elementos anteriores si es necesario
        if remove and self.pbline is not None:
            self.spectrum_widget.removeItem(self.pbline)
            self.pbline = None

        # Determinar datos según el contexto
        if self.ix is None or self.iy is None:
            return

        if self.specres:
            # Modo residual
            y_data = self.res[:, self.iy, self.ix]
            x_data = self.wl
            factor = self.fff
        else:
            # Modo normal
            if self.fitscom and "Synthetic" not in str(self.fitscom):
                if (self.wl2[0] <= self.wmax <= self.wl2[-1]):
                    y_data = self.dat2[:, self.iy, self.ix]
                    x_data = self.wl2
                    factor = self.fff2
                else:
                    y_data = self.dat[:, self.iy, self.ix]
                    x_data = self.wl
                    factor = self.fff
            else:
                y_data = self.dat[:, self.iy, self.ix]
                x_data = self.wl
                factor = self.fff

        # Calcular valores de la banda
        y_fill = factor * y_data.max() * self.fp

        # Crear elemento gráfico con relleno
        self.pbline = pg.PlotDataItem(
            x_data,
            y_fill,
            pen=pg.mkPen('g', width=1.5),
            fillLevel=0,
            brush=pg.mkBrush('g', alpha=0.25)
        )

        # Añadir al widget
        self.spectrum_widget.addItem(self.pbline)

        # Ajustar límites de los ejes
        self.spectrum_widget.setXRange(self.awlmin, self.awlmax)
        if self.fmin is not None and self.fmax is not None:
            self.spectrum_widget.setYRange(self.fmin, self.fmax)

    def ChangeFilter(self, key=None):
        """
        Cambia el filtro espectral y actualiza la visualización en PyQtGraph.
        Debe llamarse desde eventFilter o keyPressEvent, pasando la tecla presionada.
        """
        # Si se llama desde eventFilter, 'key' es un int de Qt.Key_*
        # Si se llama desde otro sitio, puede ser un string
        if key is None:
            return

        # Mapea Qt.Key_* a string si es necesario
        key_map = {
            Qt.Key_T: "t",
            Qt.Key_A: "a",
            Qt.Key_C: "c",
            Qt.Key_Shift: "T",  # Si usas Shift+T para retroceder filtro
        }
        if isinstance(key, int):
            key_str = key_map.get(key, None)
            if key_str is None:
                # Si la tecla no está mapeada, intenta obtener el carácter
                try:
                    key_str = chr(key).lower()
                except Exception:
                    return
        else:
            key_str = key

        if key_str in ["t", "T", "a", "c"]:
            if key_str == "t":
                self.ifil += 1
                if self.ifil >= self.nfil:
                    self.ifil = 0
            if key_str == "T":
                self.ifil -= 1
                if self.ifil < 0:
                    self.ifil = self.nfil - 1
            if key_str == "a":
                self.cfilter = not self.cfilter
            if key_str == "c":
                self.remove_cont = not self.remove_cont
            if key_str != "c":
                self.gdl = 0

            # Recalcular filtro e imagen
            self.set_fff(verb=True, dl=self.gdl)
            self.updatePatch()
            self.updatePassBand(remove=True)

            # Refrescar widgets de PyQtGraph (no hace falta .draw())
            self.update_spaxel_view()
            self.update_spectrum_view()

    def SpectraViewer(self, event):
        # Obtener posición del mouse en coordenadas de datos
        pos = event.scenePos()
        view_pos = self.spaxel_widget.getPlotItem().vb.mapSceneToView(pos)
        x, y = view_pos.x(), view_pos.y()

        # Convertir a píxeles del array
        ix, iy = self.res2pix(x, y)
        ix, iy = int(ix), int(iy)

        # Verificar límites y datos válidos
        if (0 <= ix < self.xx) and (0 <= iy < self.yy):
            # Mostrar información en consola (podrías usar QLabel en la UI)
            print(f"(RA,DEC) = ({x:.2f},{y:.2f}) | (x,y) = ({ix},{iy}) | z = {self.color[iy, ix]:.2f}")

            # Manejar selección con tecla 'd'
            if event.button() == Qt.RightButton:  # Usar Qt.MouseButton para mayor claridad
                if (ix, iy) in self.list:
                    self.remove_selected_spaxel(ix, iy)

            # Click izquierdo: selección normal
            elif event.button() == Qt.LeftButton:
                self.handle_left_click(ix, iy)

            # Actualizar visualización si estamos en modo espectro
            if self.mode:
                self.ix, self.iy = ix, iy
                self.PlotSpec()
                self.update_spectrum_view()

    def handle_left_click(self, ix, iy):
        if (ix, iy) not in self.list:
            # Convertir a coordenadas de resolución y crear rectángulo
            x_res, y_res = self.pix2res(ix, iy)
            xr, yr = PRectangle(x_res, y_res, self.sr)

            # Crear elemento gráfico
            rect_item = pg.PolygonROI(
                list(zip(xr, yr)),
                pen=pg.mkPen(self.cc, width=self.clw),
                movable=False
            )
            self.spaxel_widget.addItem(rect_item)

            # Guardar referencia
            self.list.append((ix, iy))
            self.pat.append(rect_item)

    def remove_selected_spaxel(self, ix, iy):
        idx = self.list.index((ix, iy))
        item = self.pat.pop(idx)
        self.spaxel_widget.removeItem(item)
        self.list.pop(idx)

    def PlotSpec(self):
        if self.ix is None or self.iy is None:
            return

        # Limpiar espectro previo y leyenda
        self.spectrum_widget.clear()
        if hasattr(self, 'legend'):
            self.spectrum_widget.removeItem(self.legend)
        self.legend = self.spectrum_widget.addLegend()

        # Obtener espectro, error y flag
        self.getSpec(self.ix, self.iy)
        label = self.slabel

        # Añadir info de zona si aplica
        if self.zones is not None and not isinstance(self.zones[self.iy, self.ix], np.ma.core.MaskedConstant):
            label = f"{label} (#{self.zones[self.iy, self.ix]})"

        # Espectro principal
        pen_main = pg.mkPen(self.cspec, width=self.lspec)
        curve_main = self.spectrum_widget.plot(self.wl, self.spec, pen=pen_main, name=label)

        # Espectro de comparación (si existe y no estamos en modo residual)
        if self.fitscom is not None and not self.specres:
            if self.dat2.ndim == 3:
                pen_comp = pg.mkPen(self.ccom, width=self.lcom)
                curve_comp = self.spectrum_widget.plot(self.wl2, self.dat2[:, self.iy, self.ix], pen=pen_comp,
                                                       name=self.fitscom)
                # Barras de error de comparación
                if self.errcom and self.err2 is not None:
                    error_bar_comp = pg.ErrorBarItem(
                        x=self.wl2,
                        y=self.dat2[:, self.iy, self.ix],
                        top=self.err2[:, self.iy, self.ix],
                        bottom=self.err2[:, self.iy, self.ix],
                        beam=0.5,
                        pen=pg.mkPen('grey')
                    )
                    self.spectrum_widget.addItem(error_bar_comp)
            elif self.dat2.ndim == 1:
                pen_comp = pg.mkPen(self.ccom, width=self.lcom)
                curve_comp = self.spectrum_widget.plot(self.wl2, self.dat2, pen=pen_comp, name=self.fitscom)
                if self.errcom and self.err2 is not None:
                    error_bar_comp = pg.ErrorBarItem(
                        x=self.wl2,
                        y=self.dat2,
                        top=self.err2,
                        bottom=self.err2,
                        beam=0.5,
                        pen=pg.mkPen('grey')
                    )
                    self.spectrum_widget.addItem(error_bar_comp)

        # Barras de error del espectro principal
        if self.errcom and self.espec is not None:
            error_bar_main = pg.ErrorBarItem(
                x=self.wl,
                y=self.spec,
                top=self.espec,
                bottom=self.espec,
                beam=0.5,
                pen=pg.mkPen('grey')
            )
            self.spectrum_widget.addItem(error_bar_main)

        # Flags o perr/pres
        if self.perr is not None and not self.specres:
            pen_flag = pg.mkPen(self.cflag, width=self.lflag)
            self.spectrum_widget.plot(self.wl, self.perr[:, self.iy, self.ix], pen=pen_flag, name="Flag")
        if self.pres is not None and self.specres:
            pen_flag = pg.mkPen(self.cflag, width=self.lflag)
            self.spectrum_widget.plot(self.wl, self.pres[:, self.iy, self.ix], pen=pen_flag, name="Residual Flag")

        # Banda de paso (filtro)
        self.updatePassBand(remove=False)

        # Ajustar límites de los ejes
        if hasattr(self, "awlmin") and hasattr(self, "awlmax"):
            self.spectrum_widget.setXRange(self.awlmin, self.awlmax)
        if hasattr(self, "fmin") and hasattr(self, "fmax"):

            self.spectrum_widget.setYRange(self.fmin, self.fmax)

        # Título
        self.spectrum_widget.setTitle(label, color='k', size='14pt')

    def getSpec(self, ix, iy):
        """
        Obtiene el espectro, error y flag para el spaxel (ix, iy) y actualiza la etiqueta.
        Compatible con PyQt5/PyQtGraph.
        """
        if ix is None or iy is None:
            self.spec = None
            self.espec = None
            self.fspec = None
            self.slabel = ""
            return

        # Determina si mostrar residuales o datos originales
        if getattr(self, "specres", False):
            self.spec = self.res[:, iy, ix] if hasattr(self, "res") and self.res is not None else None
        else:
            self.spec = self.dat[:, iy, ix] if hasattr(self, "dat") and self.dat is not None else None

        # Error y flag asociados
        self.espec = self.err[:, iy, ix] if hasattr(self, "err") and self.err is not None else None
        self.fspec = self.flag[:, iy, ix] if hasattr(self, "flag") and self.flag is not None else None

        # Etiqueta para el espectro mostrado
        self.slabel = f"Spaxel ID = {ix:0{self.fx}d} , {iy:0{self.fy}d}"

    def PlotPycassoIntSpec(self):
        # Limpiar widget de espectro
        self.spectrum_widget.clear()
        if hasattr(self, 'legend'):
            self.spectrum_widget.removeItem(self.legend)

        # Configurar título y leyenda
        strid = f"Pycasso Integrated Spectra (Total: {np.max(self.zones)} zones)"
        self.spectrum_widget.setTitle(strid)
        self.legend = self.spectrum_widget.addLegend()

        # Obtener datos del espectro integrado
        if self.fobj.pycasso == 1:
            self.spec = self.fobj.K.integrated_f_obs.copy() * self.fo
            self.espec = self.fobj.K.integrated_f_err.copy() * self.fo
            self.fspec = self.fobj.K.integrated_f_flag.copy().astype(bool)
        else:
            self.spec = self.fobj.K.integ_f_obs.copy() * self.fobj.K.flux_unit * self.fo
            self.espec = self.fobj.K.integ_f_err.copy() * self.fobj.K.flux_unit * self.fo
            self.fspec = self.fobj.K.integ_f_flag.copy().astype(bool)

        # Manejar valores NaN en errores
        self.espec[self.fspec] = np.nan

        # Plotear espectro principal
        main_pen = pg.mkPen(self.cspec, width=self.lspec)
        self.spectrum_widget.plot(
            self.wl,
            self.spec,
            pen=main_pen,
            name=strid,
            symbol='+',
            symbolSize=10
        )

        # Plotear datos de comparación si existen
        if self.fitscom and self.fobj2 and self.fobj2.K:
            if self.fobj.pycasso == 1:
                cspec = self.fobj2.K.integrated_f_obs * self.fc * self.fobj.K.flux_unit
                cspec[self.fobj2.K.integrated_f_flag.astype(bool)] = np.nan
            else:
                cspec = self.fobj2.K.integ_f_obs * self.fc
                cspec[self.fobj2.K.integ_f_flag.astype(bool)] = np.nan

            comp_pen = pg.mkPen(self.ccom, width=self.lcom)
            self.spectrum_widget.plot(
                self.wl2,
                cspec,
                pen=comp_pen,
                name=self.fitscom,
                symbol='o',
                symbolSize=8
            )

        # Plotear espectros sintéticos si corresponde
        if self.syncom:
            if self.fobj.pycasso == 1:
                syn_spec = self.fobj.K.integrated_f_syn.copy() * self.fc
            else:
                syn_spec = self.fobj.K.integ_f_syn.copy() * self.fc * self.fobj.K.flux_unit

            self.spectrum_widget.plot(
                self.wl,
                syn_spec,
                pen=comp_pen,
                name="Synthetic"
            )

        # Añadir barras de error
        if self.espec is not None and self.errcom:
            error_bar = pg.ErrorBarItem(
                x=self.wl,
                y=self.spec,
                top=self.espec,
                bottom=self.espec,
                beam=0.5,
                pen=pg.mkPen('grey', width=1)
            )
            self.spectrum_widget.addItem(error_bar)

        # Añadir línea de flags
        if self.fobj.pycasso == 1:
            fintspec = self.fobj.K.integrated_f_obs.copy()
        else:
            fintspec = self.fobj.K.integ_f_obs.copy() * self.fobj.K.flux_unit

        fintspec[~self.fspec] = np.nan
        flag_pen = pg.mkPen(self.cflag, width=self.lflag)
        self.spectrum_widget.plot(
            self.wl,
            fintspec,
            pen=flag_pen,
            name="Flags"
        )

        # Añadir banda de paso
        y_fill = self.fff * self.spec.max() * self.fp
        fill_item = pg.PlotDataItem(
            self.wl,
            y_fill,
            pen='g',
            fillLevel=0,
            brush=(0, 255, 0, 50)
        )
        self.spectrum_widget.addItem(fill_item)

        # Ajustar límites de los ejes
        self.spectrum_widget.setXRange(self.awlmin, self.awlmax)
        self.spectrum_widget.setYRange(self.fmin, self.fmax)

    def GetSpectraInfo(self, event):
        """
        Muestra información del espectro seleccionado y resalta el spaxel correspondiente en PyQtGraph.
        """
        if self.view_pintspec:
            return

        # Oculta/elimina los círculos anteriores
        for item in self.cir:
            if hasattr(item, "setVisible"):
                item.setVisible(False)
            elif hasattr(item, "hide"):
                item.hide()
            else:
                try:
                    self.spaxel_widget.removeItem(item)
                except Exception:
                    pass
        self.cir = []

        # Obtener el label y color del espectro seleccionado
        try:
            label = event.artist.opts.get('name', None)
            color = event.artist.opts.get('pen', self.cspec)
            if hasattr(color, 'color'):
                color = color.color()
        except Exception:
            label = None
            color = self.cspec

        if label is None:
            return

        stit = label
        # Extraer el ID del label si está en el formato esperado
        if "=" in label:
            stit = label.split("=")[1].split("(")[0].strip()

        # Si el label corresponde a un spaxel individual (no integrado)
        if str(stit) != self.sint and "," in stit:
            try:
                ilx, ily = map(int, stit.split(","))
            except Exception:
                return
            lx, ly = self.pix2res(ilx, ily)
            xr, yr = PRectangle(lx, ly, self.sr)
            # Crear un polígono que marque el spaxel seleccionado
            poly = pg.PlotDataItem(xr, yr, pen=pg.mkPen(color, width=self.slw), brush=pg.mkBrush(color), fillLevel=0,
                                   name="Selected Spaxel")
            self.spaxel_widget.addItem(poly)
            self.cir.append(poly)
            # Actualizar los datos del espectro seleccionado
            self.spec = self.dat[:, ily, ilx]
            self.espec = self.err[:, ily, ilx] if self.err is not None else None
            self.fspec = self.flag[:, ily, ilx] if self.flag is not None else None
            if self.zones is not None:
                stit = f"{stit} (#{self.zones[ily, ilx]})"
        else:
            self.spec = self.intspec
            self.espec = self.eintspec

        # Actualizar el título del espectro
        title = f"Spaxel ID = {stit}"
        self.spectrum_widget.setTitle(title, color=color, size='12pt')

        # Refrescar la visualización del espectro
        self.update_spectrum_view()

    def PassBandPress(self, event):
        """Maneja el evento de clic del ratón sobre la banda de paso en PyQtGraph"""
        if self.pbline is None or self.pband is None:
            return

        # Obtener posición del mouse en coordenadas de datos
        pos = event.scenePos()
        mouse_point = self.spectrum_widget.getViewBox().mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()

        # Verificar si el clic está dentro de la banda de paso
        x_data = self.pbline.xData
        y_data = self.pbline.yData
        if len(x_data) == 0 or len(y_data) == 0:
            return

        # Encontrar el punto más cercano en los datos
        idx = np.abs(np.array(x_data) - x).argmin()
        y_pb = y_data[idx]

        # Comprobar si el clic está dentro del área de la banda (0 < y < y_pb)
        if 0 <= y <= y_pb:
            # Inicializar variables para arrastre
            self.dragging_pband = True
            self.drag_start_x = x
            self.original_pbline_x = np.array(x_data)  # Copia de seguridad de los datos originales
            self.original_pbline_y = np.array(y_data)

            # Guardar posición inicial para cálculos de desplazamiento
            self.pressevent_pos = (x, y)

    def PassBandMove(self, event):
        if not hasattr(self, 'drag_start_x') or self.pbline is None:
            return

        # Obtener posición actual del mouse en coordenadas de datos
        current_pos = self.spectrum_widget.getViewBox().mapSceneToView(event.scenePos())
        current_x = current_pos.x()

        # Calcular desplazamiento en el dominio espectral
        delta_x = current_x - self.drag_start_x

        # Actualizar posición de la línea de la banda de paso
        new_x = self.initial_pbline_x + delta_x
        self.pbline.setPos(new_x)

        # Actualizar región rellena asociada
        self.updatePassBand()

    def PassBandRelease(self, event):
        """Maneja la liberación del mouse después de arrastrar la banda de paso"""
        if not hasattr(self, 'dragging_pb') or not self.dragging_pb:
            return

        # Finalizar el arrastre
        self.dragging_pb = False

        # Actualizar posición global del filtro
        self.gdl += self.delta_pb
        self.set_fff(verb=False, dl=self.gdl)

        # Actualizar visualizaciones
        self.updatePassBand()
        self.update_spaxel_view()

        # Actualizar colorbar si es necesario
        if self.iclm:
            if hasattr(self, 'hist') and self.hist is not None:
                self.hist.setLevels(*get_min_max(self.color))

        # Limpiar variables temporales
        del self.delta_pb

    def ChangeSpaxelViewer(self):
        """
        Permite al usuario cambiar la propiedad Pycasso visualizada en el mapa de spaxels.
        """
        if self.K is None:
            QMessageBox.warning(self, "Error", "*** Non Pycasso Object found! ***")
            return

        # Mostrar diálogo para pedir la propiedad
        prop, ok = QInputDialog.getText(
            self,
            "Pycasso Property",
            "***** Pycasso Property *****\nInput example: McorSD\nEnter property (Enter to abort):"
        )
        if not ok or len(prop.strip()) == 0:
            return

        # Buscar y cargar la propiedad
        newprop = self.GetPycassoProp(prop.strip())
        if newprop is not None:
            self.color = newprop
            self.updateAx1(False)
        else:
            QMessageBox.information(
                self,
                "Nada que mostrar",
                f'Nothing to plot! Property "{prop.strip()}" not found'
            )

    def updateAx1(self, color=True):
        """
        Actualiza la vista principal de spaxels en PyQtGraph.
        """
        # Si se solicita, recalcula el mapa de color con el filtro actual
        if color:
            self.set_fff(verb=False, dl=self.gdl)

        # Limpiar el widget de spaxels antes de actualizar
        self.spaxel_widget.clear()

        # Normalización y límites de color
        self.cmin, self.cmax = get_min_max(self.color)
        levels = (self.cmin, self.cmax)

        # Crear el ImageItem (equivalente a imshow)
        img = pg.ImageItem()
        img.setImage(self.color.T, levels=levels, opacity=self.palpha)  # Transponer para orientación matplotlib

        # Ajustar el rectángulo espacial (extent)
        if hasattr(self, 'ext'):
            img.setRect(pg.QtCore.QRectF(
                self.ext[0], self.ext[2],
                self.ext[1] - self.ext[0], self.ext[3] - self.ext[2]
            ))
            self.spaxel_widget.setXRange(self.ext[0], self.ext[1])
            self.spaxel_widget.setYRange(self.ext[2], self.ext[3])

        self.spaxel_widget.addItem(img)

        # Título
        title = pg.TextItem(self.bname_fits, anchor=(0.5, 1.0), color='k')
        self.spaxel_widget.addItem(title)
        title.setPos((self.ext[0] + self.ext[1]) / 2, self.ext[3])

        # Barra de color (colorbar)
        if getattr(self, 'colorbar', False):
            # Usa ColorBarItem o HistogramLUTItem según tu preferencia
            try:
                hist = pg.HistogramLUTItem()
                hist.setImageItem(img)
                hist.gradient.loadPreset('viridis')
                self.spaxel_widget.scene().addItem(hist)
            except Exception:
                pass

        # DS9-like dynamic range de colormap
        if self.iclm:
            # Si tienes una clase IntColorMap adaptada a PyQtGraph, úsala aquí
            # Si no, puedes dejarlo como placeholder o usar el histograma interactivo
            pass

        # Superponer círculos de spaxels (si corresponde)
        if hasattr(self, 'x') and hasattr(self, 'y') and hasattr(self, 'radius'):
            for xi, yi in zip(self.x, self.y):
                circ = pg.QtWidgets.QGraphicsEllipseItem(
                    xi - self.radius, yi - self.radius,
                    2 * self.radius, 2 * self.radius
                )
                circ.setPen(pg.mkPen('gray', width=1))
                circ.setBrush(pg.mkBrush(None))
                self.spaxel_widget.addItem(circ)

        # Actualizar la referencia al ImageItem principal
        self.img = img

    def GetPycassoProp(self, prop):
        """
        Devuelve la propiedad espacializada de Pycasso si existe y tiene la forma adecuada.
        """
        if self.K is None:
            return None

        # Listar atributos públicos de K
        lprop = [item for item in dir(self.K) if not item.startswith("_")]
        sprop = None

        # Buscar coincidencia directa
        if prop in lprop:
            sprop = prop

        # Para versiones antiguas (<2.0.0), buscar con sufijo __yx
        if Version(self.pversion) < Version("2.0.0"):
            if (prop + "__yx") in lprop:
                sprop = prop + "__yx"

        if sprop is None:
            return None

        # Obtener la propiedad de forma segura
        fprop = getattr(self.K, sprop, None)
        if fprop is None:
            return None

        # Para versiones nuevas, espacializar si corresponde
        if Version(self.pversion) >= Version("2.0.0"):
            if getattr(self.K, "hasSegmentationMask", False):
                fprop = self.K.spatialize(fprop, extensive=False)
            if hasattr(fprop, "ndim") and fprop.ndim == 3:
                fprop = fprop.sum(axis=0)

        # Comprobar que la forma es compatible con el mapa de datos
        nz, ny, nx = self.dat.shape
        if hasattr(fprop, "shape") and (ny, nx) == fprop.shape:
            return fprop
        else:
            return None

    def GetPycassoRes(self, key=None):
        """
        Alterna entre mostrar el mapa de datos y el mapa de residuos Pycasso en el visor de spaxels.
        """
        # Por ejemplo: if key == Qt.Key_R: self.GetPycassoRes(key)
        if key not in [Qt.Key_R, "R"]:
            return

        self.rshow = not self.rshow

        if self.K is None:
            QMessageBox.warning(self, "Error", "*** Non Pycasso Object found! ***")
            return

        if self.rshow:
            # Mostrar mapa de residuos
            print(">>> Residual Map")
            self.dcolor = self.K.res
            self.plotResidualMap()
        else:
            # Volver al mapa de datos
            print(">>> Data Map")
            self.dcolor = self.dat
            self.updateAx1()

    def plotResidualMap(self):
        """
        Muestra el mapa 2D de residuos y el espectro integrado de residuos usando PyQtGraph.
        """
        # Crear ventana de residuos
        self.residual_window = QWidget()
        self.residual_window.setWindowTitle("2D Residual Map")
        self.residual_window.resize(900, 700)
        main_layout = QVBoxLayout(self.residual_window)

        # ----- Mapa 2D de residuos -----
        pg.setConfigOption('imageAxisOrder', 'row-major')
        img_widget = pg.PlotWidget()
        img_widget.setBackground('w')
        img_widget.setTitle("2D Residual Map")
        img_widget.setLabel('left', "# Zone")
        img_widget.setLabel('bottom', "Wavelength (Å)")

        # Extraer datos y límites
        awlmin, awlmax = self.wl[0], self.wl[-1]
        zres = self.K.zres.T  # (zones, lambda)
        extent = [awlmin, awlmax, 0.5, zres.shape[0] + 0.5]

        # Crear imagen de residuos
        img_item = pg.ImageItem(zres)
        img_item.setLookupTable(pg.colormap.get('bwr').getLookupTable(0.0, 1.0, 256))
        img_item.setLevels([np.nanmin(zres), np.nanmax(zres)])
        img_widget.addItem(img_item)
        img_widget.setLimits(xMin=awlmin, xMax=awlmax, yMin=0.5, yMax=zres.shape[0] + 0.5)
        img_widget.setAspectLocked(False)
        img_widget.setRange(xRange=(awlmin, awlmax), yRange=(0.5, zres.shape[0] + 0.5))

        # Añadir colorbar
        hist = pg.HistogramLUTItem()
        hist.setImageItem(img_item)
        hist.gradient.loadPreset('bwr')
        img_widget.scene().addItem(hist)

        # Layout para mapa y colorbar
        map_layout = QHBoxLayout()
        map_layout.addWidget(img_widget)
        main_layout.addLayout(map_layout)

        # ----- Espectro integrado de residuos -----
        tres_widget = pg.PlotWidget()
        tres_widget.setBackground('w')
        tres_widget.setLabel('left', "Integrated Residual")
        tres_widget.setLabel('bottom', "Wavelength (Å)")
        tres_widget.setXRange(awlmin, awlmax)
        tres_widget.setYRange(-0.1, 0.1)
        tres_widget.plot(self.wl, self.K.tres, pen=pg.mkPen('b', width=2))
        main_layout.addWidget(tres_widget)

        # Instalar el filtro de eventos
        self.installEventFilter(self)

        # Mostrar ventana
        self.residual_window.show()

        # (Opcional) actualizar el mapa de spaxels principal
        self.updateAx1()

    def plotResidualMap(self):
        self.res_viewer = ResidualViewer()
        self.res_viewer.setup_data(self.wl, self.zones, self.K.zres, self.K.tres)
        self.res_viewer.show()

    def selectZone(self):
        """
        Permite al usuario seleccionar una zona Pycasso y resalta los spaxels correspondientes en el visor.
        """
        if self.K is None:
            QMessageBox.warning(self, "Error", "*** Non Pycasso Object found! ***")
            return

        # Solicitar zona al usuario
        zn, ok = QInputDialog.getText(
            self,
            "Select Pycasso Zone",
            "***** Select Pycasso Zone *****\nInput example: 3\nEnter zone ID (Enter to abort):"
        )
        if not ok or len(zn.strip()) == 0:
            return

        try:
            zone_id = int(zn.strip())
        except Exception:
            QMessageBox.warning(self, "Error", "Invalid zone ID.")
            return

        lzy, lzx = np.where(self.zones == zone_id)
        if lzy.size < 1 or lzx.size < 1:
            QMessageBox.information(self, "No Zone", f"*** No available ZONE with ID: {zone_id} ***")
            return

        # Eliminar selecciones previas
        for item in getattr(self, "pat", []):
            if isinstance(item, pg.QtWidgets.QGraphicsItem):
                self.spaxel_widget.removeItem(item)
            elif isinstance(item, list) and len(item) > 0 and hasattr(item[0], "setVisible"):
                item[0].setVisible(False)
        self.list = []
        self.pat = []

        # Resaltar los spaxels de la zona seleccionada
        for zy, zx in zip(lzy, lzx):
            if (zx, zy) not in self.list:
                ir, id_ = self.pix2res(zx, zy)
                xr, yr = PRectangle(ir, id_, self.sr)
                # Crear un polígono para resaltar el spaxel
                poly = pg.PlotDataItem(
                    xr, yr,
                    pen=pg.mkPen(self.cc, width=self.clw),
                    brush=pg.mkBrush(self.cc if self.cf else None),
                    fillLevel=0
                )
                self.spaxel_widget.addItem(poly)
                self.list.append((zx, zy))
                self.pat.append(poly)

        # Actualizar la vista de spaxels si es necesario
        self.update_spaxel_view()

    def FitSpec(self, key=None):
        """
        Permite seleccionar y lanzar el ajuste espectral usando PyRAF o PySPECKIT desde PyQt5/PyQtGraph.
        """
        # Debe llamarse desde eventFilter o keyPressEvent, pasando la tecla presionada (Qt.Key_I, Qt.Key_X, etc.)
        if not self.fitspec and (key in [Qt.Key_I, Qt.Key_X, "i", "x"]):
            QMessageBox.warning(self, "Fitting",
                                '*** You need module "pyspeckit" or "pyraf" to use fitting features!!! ***')
            return

        # Cambiar modo de fitting
        if key in [Qt.Key_I, "i"]:
            if self.pyraf and not self.pyspec:
                self.fitspec_mode = 0
                QMessageBox.information(self, "Fitting", ">>> Only PyRAF module available")
            elif not self.pyraf and self.pyspec:
                self.fitspec_mode = 1
                QMessageBox.information(self, "Fitting", ">>> Only PySPECKIT module available")
            else:
                self.fitspec_mode += 1
                if self.fitspec_mode % 2 == 0:
                    QMessageBox.information(self, "Fitting", ">>> PyRAF fitting mode selected")
                else:
                    QMessageBox.information(self, "Fitting", ">>> PySPECKIT fitting mode selected")
            return

        # Lanzar el ajuste sobre el primer spaxel seleccionado
        if key in [Qt.Key_X, "x"] and len(self.list) > 0:
            ix, iy = self.list[0]
            self.getSpec(ix, iy)
            if self.fitspec_mode % 2 == 0:
                # PyRAF mode
                try:
                    import matplotlib.pyplot as plt
                    if plt.get_backend() != "TkAgg":
                        QMessageBox.warning(self, "Fitting",
                                            '*** You need to set backend to "TkAgg" if you want to use PyRAF interactive fitting ***')
                        return
                    sname = "%s_%s" % (".".join(self.name_fits.split(".")[0:-1]), "%s_%s" % (ix, iy))
                    tmpfits = tmpName(prefix="tmp_%s" % sname)
                    convert2iraf_spec(tmpfits, self.wl, self.spec, title=sname)
                    QMessageBox.information(self, "Fitting", f">>> Spectrum ({self.idl}) of {self.name_fits}")

                    import pyraf
                    pyraf.iraf.splot(tmpfits)
                    import os
                    if os.path.exists(tmpfits):
                        os.remove(tmpfits)
                except Exception as e:
                    QMessageBox.critical(self, "Fitting", f"Error running PyRAF fitting: {e}")
            else:
                # PySPECKIT mode
                try:
                    import pyspeckit
                    sp = pyspeckit.Spectrum(
                        xarr=self.wl, data=self.spec, error=self.espec, header=self.hd
                    )
                    sp.plotter()
                    sp.plotter.axis.set_xlabel(r"Wavelength $(\AA)$")
                    sp.plotter.axis.set_ylabel(r"Flux $(\mathrm{erg/s/cm^2/\AA})$")
                    sp.plotter.axis.set_title(f"{sp.plotter.title} ({ix}, {iy})")
                    import matplotlib.pyplot as plt
                    plt.show()
                except Exception as e:
                    QMessageBox.critical(self, "Fitting", f"Error running PySPECKIT fitting: {e}")

    def WindowManager(self):
        """Muestra el administrador de ventanas y controles interactivos en PyQt5."""
        manager_window = QWidget()
        manager_window.setWindowTitle(self.winman_label)
        manager_window.resize(*[int(x * 100) for x in self.winman_size])

        main_layout = QVBoxLayout(manager_window)

        # ---- Spaxel Properties ----
        spaxel_group = QGroupBox("Spaxel Properties")
        spaxel_layout = QVBoxLayout(spaxel_group)
        spaxel_alpha_slider = QSlider(Qt.Horizontal)
        spaxel_alpha_slider.setRange(0, 100)
        spaxel_alpha_slider.setValue(int(self.palpha * 100))
        spaxel_alpha_label = QLabel(f"Alpha: {self.palpha:.2f}")
        spaxel_layout.addWidget(spaxel_alpha_label)
        spaxel_layout.addWidget(spaxel_alpha_slider)
        main_layout.addWidget(spaxel_group)

        # ---- Circle Selector Properties ----
        circle_group = QGroupBox("Spaxel Selector Properties")
        circle_layout = QGridLayout(circle_group)
        circle_lw_slider = QSlider(Qt.Horizontal)
        circle_lw_slider.setRange(1, 10)
        circle_lw_slider.setValue(self.clw)
        circle_lw_label = QLabel(f"Linewidth: {self.clw}")
        circle_alpha_slider = QSlider(Qt.Horizontal)
        circle_alpha_slider.setRange(0, 100)
        circle_alpha_slider.setValue(int(self.ca * 100))
        circle_alpha_label = QLabel(f"Alpha: {self.ca:.2f}")
        circle_fill_btn = QPushButton(self.clab)
        circle_layout.addWidget(circle_lw_label, 0, 0)
        circle_layout.addWidget(circle_lw_slider, 0, 1)
        circle_layout.addWidget(circle_alpha_label, 1, 0)
        circle_layout.addWidget(circle_alpha_slider, 1, 1)
        circle_layout.addWidget(circle_fill_btn, 2, 0, 1, 2)
        main_layout.addWidget(circle_group)

        # ---- Spectra Selector Properties ----
        spectra_group = QGroupBox("Spectra → Spaxel Identifier Properties")
        spectra_layout = QGridLayout(spectra_group)
        spectra_lw_slider = QSlider(Qt.Horizontal)
        spectra_lw_slider.setRange(1, 10)
        spectra_lw_slider.setValue(self.slw)
        spectra_lw_label = QLabel(f"Linewidth: {self.slw}")
        spectra_alpha_slider = QSlider(Qt.Horizontal)
        spectra_alpha_slider.setRange(0, 100)
        spectra_alpha_slider.setValue(int(self.sa * 100))
        spectra_alpha_label = QLabel(f"Alpha: {self.sa:.2f}")
        spectra_fill_btn = QPushButton(self.slab)
        spectra_layout.addWidget(spectra_lw_label, 0, 0)
        spectra_layout.addWidget(spectra_lw_slider, 0, 1)
        spectra_layout.addWidget(spectra_alpha_label, 1, 0)
        spectra_layout.addWidget(spectra_alpha_slider, 1, 1)
        spectra_layout.addWidget(spectra_fill_btn, 2, 0, 1, 2)
        main_layout.addWidget(spectra_group)

        # ---- Save File Options ----
        save_group = QGroupBox("Save File Options")
        save_layout = QGridLayout(save_group)
        integrated_btn = QPushButton(self.intlab)
        individual_btn = QPushButton(self.indlab)
        txt_btn = QPushButton(self.txtlab)
        fits_btn = QPushButton(self.fitlab)
        save_layout.addWidget(QLabel("Spectra Type:"), 0, 0)
        save_layout.addWidget(integrated_btn, 1, 0)
        save_layout.addWidget(individual_btn, 2, 0)
        save_layout.addWidget(QLabel("File Type:"), 0, 1)
        save_layout.addWidget(txt_btn, 1, 1)
        save_layout.addWidget(fits_btn, 2, 1)
        main_layout.addWidget(save_group)

        # ---- Conexión de señales ----
        def update_spaxel_alpha(val):
            self.palpha = val / 100.0
            spaxel_alpha_label.setText(f"Alpha: {self.palpha:.2f}")
            self.updateAx1()

        def update_circle_lw(val):
            self.clw = val
            circle_lw_label.setText(f"Linewidth: {self.clw}")
            for item in self.pat:
                if hasattr(item, "setPen"):
                    pen = item.pen()
                    pen.setWidth(self.clw)
                    item.setPen(pen)
            self.updateAx1()

        def update_circle_alpha(val):
            self.ca = val / 100.0
            circle_alpha_label.setText(f"Alpha: {self.ca:.2f}")
            for item in self.pat:
                if hasattr(item, "setOpacity"):
                    item.setOpacity(self.ca)
            self.updateAx1()

        def toggle_circle_fill():
            self.cf = not self.cf
            circle_fill_btn.setText("Fill On" if self.cf else "Fill Off")
            for item in self.pat:
                if hasattr(item, "setBrush"):
                    item.setBrush(QBrush(QColor(self.cc)) if self.cf else QBrush())
            self.updateAx1()

        def update_spectra_lw(val):
            self.slw = val
            spectra_lw_label.setText(f"Linewidth: {self.slw}")
            for item in self.cir:
                if hasattr(item, "setPen"):
                    pen = item.pen()
                    pen.setWidth(self.slw)
                    item.setPen(pen)
            self.updateAx1()

        def update_spectra_alpha(val):
            self.sa = val / 100.0
            spectra_alpha_label.setText(f"Alpha: {self.sa:.2f}")
            for item in self.cir:
                if hasattr(item, "setOpacity"):
                    item.setOpacity(self.sa)
            self.updateAx1()

        def toggle_spectra_fill():
            self.sf = not self.sf
            spectra_fill_btn.setText("Fill On" if self.sf else "Fill Off")
            for item in self.cir:
                if hasattr(item, "setBrush"):
                    item.setBrush(QBrush(QColor(self.cflag)) if self.sf else QBrush())
            self.updateAx1()

        def toggle_integrated():
            self.integrated = not self.integrated
            integrated_btn.setText("Integrated On" if self.integrated else "Integrated Off")

        def toggle_individual():
            self.individual = not self.individual
            individual_btn.setText("Individual On" if self.individual else "Individual Off")

        def toggle_txt():
            self.txt = not self.txt
            txt_btn.setText("Txt On" if self.txt else "Txt Off")

        def toggle_fits():
            self.fits = not self.fits
            fits_btn.setText("Fits On" if self.fits else "Fits Off")

        # Conectar sliders y botones
        spaxel_alpha_slider.valueChanged.connect(update_spaxel_alpha)
        circle_lw_slider.valueChanged.connect(update_circle_lw)
        circle_alpha_slider.valueChanged.connect(update_circle_alpha)
        circle_fill_btn.clicked.connect(toggle_circle_fill)
        spectra_lw_slider.valueChanged.connect(update_spectra_lw)
        spectra_alpha_slider.valueChanged.connect(update_spectra_alpha)
        spectra_fill_btn.clicked.connect(toggle_spectra_fill)
        integrated_btn.clicked.connect(toggle_integrated)
        individual_btn.clicked.connect(toggle_individual)
        txt_btn.clicked.connect(toggle_txt)
        fits_btn.clicked.connect(toggle_fits)

        manager_window.setLayout(main_layout)
        manager_window.show()

    def Sonification(self):
        """
        Inicializa la interfaz de sonificación SoniCube en PyQt5/PyQtGraph.
        """
        if self.dsoni is None:
            QMessageBox.warning(self, "Sonification", "No se ha definido el directorio de sonificación (dsoni).")
            return

        try:
            if self.dsoni not in sys.path:
                sys.path.append(self.dsoni)
            from .sonicube import SoniCube
        except Exception as e:
            QMessageBox.critical(self, "Sonification", f"No se pudo importar SoniCube:\n{e}")
            return

        try:
            self.sc = SoniCube(
                parent=self,  # Puedes pasar self si SoniCube acepta un QWidget o None
                file=self.name_fits,
                data=self.dat,
                base_dir=self.dsoni,
                ref=(self.y_ref, self.x_ref),
            )
        except Exception as e:
            QMessageBox.critical(self, "Sonification", f"Error al inicializar SoniCube:\n{e}")
            return

        QMessageBox.information(self, "Sonification",
                                "SoniCube inicializado correctamente. Usa el modo sonificación con la tecla 'h'.")


class ResidualViewer:
    def __init__(self):
        # Configurar ventana principal
        self.residual_window = pg.GraphicsLayoutWidget()
        self.residual_window.setWindowTitle("2D Residual Map")

        # Plot principal (ax4)
        self.plot_main = self.residual_window.addPlot(title="Residual Zones", row=0, col=0)
        self.plot_main.setLabel('bottom', "Wavelength (Å)")
        self.plot_main.setLabel('left', "Zone")

        # Plot de zona (ax5)
        self.plot_zone = self.residual_window.addPlot(title="Zone Spectrum", row=1, col=0)
        self.plot_zone.setLabel('bottom', "Wavelength (Å)")
        self.plot_zone.setLabel('left', "Residual")
        self.plot_zone.setYRange(-0.1, 0.1)

        # Elementos gráficos
        self.hline = None
        self.text_label = pg.TextItem(anchor=(1, -0.5))
        self.plot_zone.addItem(self.text_label)

        # Conectar eventos del ratón
        self.plot_main.scene().sigMouseClicked.connect(self.handle_mouse_click)

    def setup_data(self, wl, zones, zres, tres):
        """Inicializar datos"""
        self.wl = wl
        self.zones = zones
        self.zres = zres
        self.tres = tres

        # Crear imagen de residuos
        self.img = pg.ImageItem(self.zres.T)
        self.img.setLookupTable(pg.colormap.get('bwr'))
        self.plot_main.addItem(self.img)
        self.plot_main.setXRange(wl[0], wl[-1])
        self.plot_main.setYRange(0.5, self.zres.shape[1] + 0.5)

    def handle_mouse_click(self, event):
        """Manejar clics en el plot principal"""
        pos = event.scenePos()
        if not self.plot_main.vb.sceneBoundingRect().contains(pos):
            return

        mouse_point = self.plot_main.vb.mapSceneToView(pos)
        y = int(mouse_point.y())

        if 0 < y <= self.zones.max():
            self.update_zone_plot(y)
        else:
            self.show_integrated()

    def update_zone_plot(self, y):
        """Actualizar plot con datos de la zona seleccionada"""
        # Actualizar línea horizontal
        if self.hline:
            self.plot_main.removeItem(self.hline)
        self.hline = pg.InfiniteLine(y, angle=0, pen='g', movable=False)
        self.plot_main.addItem(self.hline)

        # Actualizar datos
        zone_data = self.zres[:, y - 1]
        self.plot_zone.clear()
        self.plot_zone.plot(self.wl, zone_data, pen='b')

        # Actualizar texto
        self.text_label.setText(f"Zone #{y}", color='k', size='12pt')

    def show_integrated(self):
        """Mostrar espectro integrado"""
        if self.hline:
            self.plot_main.removeItem(self.hline)
            self.hline = None

        self.plot_zone.clear()
        self.plot_zone.plot(self.wl, self.tres, pen='b')
        self.text_label.setText("Integrated", color='k', size='12pt')

    def show(self):
        self.residual_window.show()