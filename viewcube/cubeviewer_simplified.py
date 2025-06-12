############################################################################
#                              VIEWCUBE                                    #
#                           SIMPLIFIED VERSION                            #
#                              PYTHON 3                                    #
#                                                                         #
# RGB@IAA ---> Last Change: 2024/10/10                                    #
############################################################################

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                           QLabel, QSpinBox, QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt, QEvent
import pyqtgraph as pg
import numpy as np
import astropy.io.fits as pyfits
import os

# Configure global pyqtgraph settings
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class CubeViewer(QWidget):
    """Main Cube Viewer widget for 3D spectral data visualization."""
    
    def __init__(self, name_fits, **kwargs):
        """Initialize the cube viewer with optional FITS file.
        
        Args:
            name_fits (str): Path to FITS file to load
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        
        # Initialize variables
        self.kwargs = kwargs
        self.name_fits = name_fits
        self.fits_data = None
        self.data = None
        self.wl = None
        self.spec = None
        self.speccom = None
        self.wlim = None
        self.flim = None
        
        # Setup UI
        self.setup_ui()
        
        # Load data if file provided
        if self.name_fits:
            self.load_data()
    
    def setup_ui(self):
        """Initialize the user interface."""
        # Main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        # Create widgets
        self.setup_spaxel_widget()
        self.setup_spectral_widget()
        
        # Add widgets to layout
        self.main_layout.addWidget(self.spaxel_widget)
        self.main_layout.addWidget(self.spectrum_widget)
        
        # Connect events
        self.spaxel_widget.scene().sigMouseClicked.connect(self.on_spaxel_click)
    
    def setup_spaxel_widget(self):
        """Configure the spaxel view widget."""
        self.spaxel_widget = pg.PlotWidget()
        self.spaxel_widget.setAspectLocked(True)
        self.spaxel_widget.showGrid(x=True, y=True)
        self.spaxel_widget.setMouseEnabled(x=True, y=True)
        self.spaxel_widget.enableAutoRange()
        self.spaxel_widget.setMenuEnabled(True)
        self.spaxel_widget.setDownsampling(auto=True, mode='peak')
        self.spaxel_widget.setLabel('left', 'Y')
        self.spaxel_widget.setLabel('bottom', 'X')
    
    def setup_spectral_widget(self):
        """Configure the spectrum view widget."""
        self.spectrum_widget = pg.PlotWidget()
        self.spectrum_widget.showGrid(x=True, y=True)
        self.spectrum_widget.setMouseEnabled(x=True, y=True)
        self.spectrum_widget.enableAutoRange()
        self.spectrum_widget.setMenuEnabled(True)
        self.spectrum_widget.setLabel('left', 'Flux')
        self.spectrum_widget.setLabel('bottom', 'Wavelength')
    
    def load_data(self):
        """Load data from FITS file."""
        if not self.name_fits:
            QMessageBox.warning(self, "Warning", "No FITS file specified")
            return
            
        try:
            # Check if file exists
            if not os.path.exists(self.name_fits):
                raise FileNotFoundError(f"File not found: {self.name_fits}")
                
            # Load FITS file
            from viewcube.utils import LoadFits
            self.fits_data = LoadFits(self.name_fits)
            
            if self.fits_data is None:
                raise ValueError("Failed to load FITS file")
                
            if not hasattr(self.fits_data, 'data') or self.fits_data.data is None:
                raise ValueError("FITS file contains no valid data")
                
            # Initialize data
            self.data = self.fits_data.data
            
            # Set wavelength if available
            if hasattr(self.fits_data, 'wave'):
                self.wl = self.fits_data.wave
            else:
                self.wl = np.arange(self.data.shape[0])
            
            # Calculate mean spectrum
            self.spec = np.nanmean(self.data, axis=(1, 2))
            
            # Update visualizations
            self.update_visualizations()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading FITS file: {str(e)}")
            self.fits_data = None
            self.data = None
            self.wl = None
            self.spec = None
    
    def update_visualizations(self):
        """Update all visualizations."""
        if not hasattr(self, 'fits_data') or self.fits_data is None:
            return
        
        self.update_spaxel_view()
        self.update_spectrum_view()
    
    def update_spaxel_view(self):
        """Update the spaxel view."""
        if not hasattr(self, 'fits_data') or self.fits_data.data is None:
            return
            
        try:
            # Clear previous view
            self.spaxel_widget.clear()
            
            # Create image
            img = pg.ImageItem()
            img_data = self.fits_data.data[0]  # First wavelength slice
            img.setImage(img_data.T)  # Transpose for correct orientation
            
            # Add image to plot
            self.spaxel_widget.addItem(img)
            
            # Auto range
            self.spaxel_widget.autoRange()
            
        except Exception as e:
            print(f"Error updating spaxel view: {e}")
    
    def update_spectrum_view(self):
        """Update the spectrum view."""
        if not hasattr(self, 'fits_data') or self.fits_data.data is None:
            return
            
        try:
            # Clear previous plot
            self.spectrum_widget.clear()
            
            # Get wavelength data
            if hasattr(self.fits_data, 'wave') and self.fits_data.wave is not None:
                x = self.fits_data.wave
            else:
                x = np.arange(self.fits_data.data.shape[0])
                
            # Calculate mean spectrum
            y = np.nanmean(self.fits_data.data, axis=(1, 2))
            
            # Apply wavelength limits if set
            if self.wlim is not None:
                mask = (x >= self.wlim[0]) & (x <= self.wlim[1])
                x = x[mask]
                y = y[mask]
            
            # Apply flux limits if set
            if self.flim is not None:
                y = np.clip(y, self.flim[0], self.flim[1])
            
            # Plot spectrum
            self.spectrum_widget.plot(x, y, pen='b', width=2)
            
        except Exception as e:
            print(f"Error updating spectrum view: {e}")
    
    def on_spaxel_click(self, event):
        """Handle mouse clicks on the spaxel view."""
        if event.button() != 1:  # Left click only
            return
            
        try:
            # Get click position in data coordinates
            pos = event.scenePos()
            view_pos = self.spaxel_widget.getPlotItem().vb.mapSceneToView(pos)
            x, y = int(round(view_pos.x())), int(round(view_pos.y()))
            
            # Check if click is within data bounds
            if (0 <= x < self.fits_data.data.shape[2] and 
                0 <= y < self.fits_data.data.shape[1]):
                # Plot spectrum for clicked spaxel
                self.plot_spaxel_spectrum(x, y)
                
        except Exception as e:
            print(f"Error handling spaxel click: {e}")
    
    def plot_spaxel_spectrum(self, x, y):
        """Plot spectrum for a specific spaxel."""
        if not hasattr(self, 'fits_data') or self.fits_data.data is None:
            return
            
        try:
            # Get wavelength data
            if hasattr(self.fits_data, 'wave') and self.fits_data.wave is not None:
                x_wave = self.fits_data.wave
            else:
                x_wave = np.arange(self.fits_data.data.shape[0])
            
            # Get spectrum for clicked spaxel
            spectrum = self.fits_data.data[:, y, x]
            
            # Update spectrum view
            self.spectrum_widget.clear()
            self.spectrum_widget.plot(x_wave, spectrum, pen='r', width=2)
            
            # Add title with coordinates
            self.spectrum_widget.setTitle(f"Spaxel ({x}, {y})")
            
        except Exception as e:
            print(f"Error plotting spaxel spectrum: {e}")
    
    def set_wavelength_limits(self, wmin, wmax):
        """Set wavelength limits for the spectrum view."""
        self.wlim = (wmin, wmax)
        self.update_spectrum_view()
    
    def set_flux_limits(self, fmin, fmax):
        """Set flux limits for the spectrum view."""
        self.flim = (fmin, fmax)
        self.update_spectrum_view()
    
    def save_spectrum(self, filename):
        """Save the current spectrum to a file."""
        if not hasattr(self, 'fits_data') or self.fits_data.data is None:
            return False
            
        try:
            # Get wavelength data
            if hasattr(self.fits_data, 'wave') and self.fits_data.wave is not None:
                x = self.fits_data.wave
            else:
                x = np.arange(self.fits_data.data.shape[0])
            
            # Get spectrum (mean across all spaxels)
            y = np.nanmean(self.fits_data.data, axis=(1, 2))
            
            # Save to file
            if filename.endswith('.txt'):
                np.savetxt(filename, np.column_stack((x, y)))
            elif filename.endswith('.fits'):
                hdu = pyfits.PrimaryHDU(np.column_stack((x, y)))
                hdu.writeto(filename, overwrite=True)
            else:
                raise ValueError("Unsupported file format. Use .txt or .fits")
                
            return True
            
        except Exception as e:
            print(f"Error saving spectrum: {e}")
            return False
