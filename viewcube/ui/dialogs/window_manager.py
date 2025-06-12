# ui/dialogs/window_manager.py
"""Gestor de ventanas que replica funcionalidad original."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from typing import Any


class WindowManager:
    """Gestor de ventanas con sliders y controles."""

    def __init__(self, figure_size=(5, 5)):
        self.figure_size = figure_size
        self.figure = None
        self.controller = None

        # Colores de botones
        self.axcolor1 = "lightgoldenrodyellow"
        self.axcolor2 = "#D0A9F5"
        self.hovercolor = "0.975"

    def show(self, controller) -> None:
        """Muestra ventana de gestión."""
        self.controller = controller

        if plt.fignum_exists(3):
            plt.figure(3)
            plt.show()
            return

        self.figure = plt.figure(3, self.figure_size)
        self.figure.set_label("Window Manager")
        self.figure.canvas.manager.set_window_title("Window Manager")

        self._setup_ui()
        plt.show()

    def _setup_ui(self) -> None:
        """Configura interfaz de usuario."""
        # Propiedades de Spaxel
        ps = gridspec.GridSpec(
            3, 2, left=0.2, bottom=0.8, top=0.95,
            wspace=0.5, hspace=0.7, right=0.8
        )

        ax = plt.subplot(ps[:2, :], frameon=False, xticks=[], yticks=[])
        ax.text(0.5, 0.5, "Spaxel Properties", ha="center")

        # Slider de alpha
        self.alpha_slider = Slider(
            plt.subplot(ps[2, :]), "Alpha", 0.0, 1.0,
            valinit=self.controller.config.get('palpha', 0.95)
        )

        # Propiedades del selector
        cs = gridspec.GridSpec(
            3, 3, left=0.2, bottom=0.55, top=0.7,
            wspace=0.7, hspace=0.5, right=0.9
        )

        ax = plt.subplot(cs[0, :], frameon=False, xticks=[], yticks=[])
        ax.text(0.5, 0.5, "Spaxel Selector Properties", ha="center")

        self.linewidth_slider = Slider(
            plt.subplot(cs[1, :-1]), "Linewidth", 0.0, 5.0,
            valinit=self.controller.config.get('clw', 1)
        )

        self.selector_alpha_slider = Slider(
            plt.subplot(cs[2, :-1]), "Alpha", 0.0, 1.0,
            valinit=self.controller.config.get('ca', 0.8)
        )

        # Botón de fill
        cf_state = self.controller.config.get('cf', False)
        color = self.axcolor2 if cf_state else self.axcolor1
        self.fill_button = Button(
            plt.subplot(cs[1:, -1]),
            "Fill On" if cf_state else "Fill Off",
            hovercolor=color, color=color
        )

        # Opciones de guardado
        svp = gridspec.GridSpec(
            3, 2, left=0.2, bottom=0.05, top=0.2,
            wspace=0.7, hspace=0.5, right=0.9
        )

        ax = plt.subplot(svp[0, 0], frameon=False, xticks=[], yticks=[])
        ax.text(0.5, 0.5, "Spectra Type", ha="center")

        ax = plt.subplot(svp[0, 1], frameon=False, xticks=[], yticks=[])
        ax.text(0.5, 0.5, "File Type", ha="center")

        # Botones de tipo de espectro
        integrated_state = self.controller.config.get('integrated', True)
        color = self.axcolor2 if integrated_state else self.axcolor1
        self.integrated_button = Button(
            plt.subplot(svp[1, 0]),
            "Integrated On" if integrated_state else "Integrated Off",
            hovercolor=color, color=color
        )

        individual_state = self.controller.config.get('individual', False)
        color = self.axcolor2 if individual_state else self.axcolor1
        self.individual_button = Button(
            plt.subplot(svp[2, 0]),
            "Individual On" if individual_state else "Individual Off",
            hovercolor=color, color=color
        )

        # Botones de tipo de archivo
        txt_state = self.controller.config.get('txt', True)
        color = self.axcolor2 if txt_state else self.axcolor1
        self.txt_button = Button(
            plt.subplot(svp[1, 1]),
            "txt On" if txt_state else "txt Off",
            hovercolor=color, color=color
        )

        fits_state = self.controller.config.get('fits', False)
        color = self.axcolor2 if fits_state else self.axcolor1
        self.fits_button = Button(
            plt.subplot(svp[2, 1]),
            "fits On" if fits_state else "fits Off",
            hovercolor=color, color=color
        )

        # Conectar eventos
        self._connect_events()

    def _connect_events(self) -> None:
        """Conecta eventos de widgets."""
        self.alpha_slider.on_changed(self._update_alpha)
        self.linewidth_slider.on_changed(self._update_linewidth)
        self.selector_alpha_slider.on_changed(self._update_selector_alpha)

        self.fill_button.on_clicked(self._toggle_fill)
        self.integrated_button.on_clicked(self._toggle_integrated)
        self.individual_button.on_clicked(self._toggle_individual)
        self.txt_button.on_clicked(self._toggle_txt)
        self.fits_button.on_clicked(self._toggle_fits)

    def _update_alpha(self, val) -> None:
        """Actualiza alpha de spaxels."""
        self.controller.config['palpha'] = val
        self.controller.spaxel_viewer.set_alpha(val)

    def _update_linewidth(self, val) -> None:
        """Actualiza grosor de línea."""
        self.controller.config['clw'] = val
        self.controller.spaxel_viewer.set_selection_linewidth(val)

    def _update_selector_alpha(self, val) -> None:
        """Actualiza alpha del selector."""
        self.controller.config['ca'] = val
        self.controller.spaxel_viewer.set_selection_alpha(val)

    def _toggle_fill(self, event) -> None:
        """Alterna fill del selector."""
        current = self.controller.config.get('cf', False)
        self.controller.config['cf'] = not current

        # Actualizar botón
        new_state = not current
        color = self.axcolor2 if new_state else self.axcolor1
        self.fill_button.color = color
        self.fill_button.hovercolor = color
        self.fill_button.ax.set_facecolor(color)
        self.fill_button.label.set_text("Fill On" if new_state else "Fill Off")

        # Actualizar visualizador
        self.controller.spaxel_viewer.set_selection_fill(new_state)
        self.figure.canvas.draw()

    def _toggle_integrated(self, event) -> None:
        """Alterna guardado integrado."""
        current = self.controller.config.get('integrated', True)
        self.controller.config['integrated'] = not current
        self._update_button_state(self.integrated_button, not current, "Integrated")

    def _toggle_individual(self, event) -> None:
        """Alterna guardado individual."""
        current = self.controller.config.get('individual', False)
        self.controller.config['individual'] = not current
        self._update_button_state(self.individual_button, not current, "Individual")

    def _toggle_txt(self, event) -> None:
        """Alterna guardado TXT."""
        current = self.controller.config.get('txt', True)
        self.controller.config['txt'] = not current
        self._update_button_state(self.txt_button, not current, "txt")

    def _toggle_fits(self, event) -> None:
        """Alterna guardado FITS."""
        current = self.controller.config.get('fits', False)
        self.controller.config['fits'] = not current
        self._update_button_state(self.fits_button, not current, "fits")

    def _update_button_state(self, button, state, label) -> None:
        """Actualiza estado visual de botón."""
        color = self.axcolor2 if state else self.axcolor1
        button.color = color
        button.hovercolor = color
        button.ax.set_facecolor(color)
        button.label.set_text(f"{label} On" if state else f"{label} Off")
        self.figure.canvas.draw()


# ui/dialogs/limits_dialog.py
"""Diálogos para configurar límites de visualización."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple


class LimitsDialog:
    """Diálogos para configurar límites de longitud de onda y flujo."""

    def __init__(self):
        self.figure = None
        self.result = None

    def get_wavelength_limits(self, current_limits: Optional[Tuple[float, float]] = None) -> Optional[
        Tuple[float, float]]:
        """Obtiene nuevos límites de longitud de onda."""
        return self._show_limits_dialog(
            "Wavelength Limits",
            "Wavelength (Å)",
            current_limits or (3000, 10000),
            (1000, 50000)
        )

    def get_flux_limits(self, current_limits: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
        """Obtiene nuevos límites de flujo."""
        return self._show_limits_dialog(
            "Flux Limits",
            "Flux",
            current_limits or (-1e-17, 1e-17),
            (-1e-15, 1e-15)
        )

    def _show_limits_dialog(self, title: str, label: str,
                            current: Tuple[float, float],
                            range_limits: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Muestra diálogo de límites genérico."""
        self.result = None

        # Crear figura
        self.figure = plt.figure(figsize=(6, 4))
        self.figure.suptitle(title)

        # Layout
        gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.2)

        # Sliders
        ax_min = plt.subplot(gs[0, :])
        ax_max = plt.subplot(gs[1, :])

        self.min_slider = Slider(
            ax_min, f"Min {label}",
            range_limits[0], range_limits[1],
            valinit=current[0]
        )

        self.max_slider = Slider(
            ax_max, f"Max {label}",
            range_limits[0], range_limits[1],
            valinit=current[1]
        )

        # Botones
        ax_apply = plt.subplot(gs[2, 0])
        ax_cancel = plt.subplot(gs[2, 1])

        apply_button = Button(ax_apply, "Apply")
        cancel_button = Button(ax_cancel, "Cancel")

        # Eventos
        apply_button.on_clicked(self._apply_limits)
        cancel_button.on_clicked(self._cancel_limits)

        # Mostrar y esperar
        plt.show()

        return self.result

    def _apply_limits(self, event) -> None:
        """Aplica límites seleccionados."""
        self.result = (self.min_slider.val, self.max_slider.val)
        plt.close(self.figure)

    def _cancel_limits(self, event) -> None:
        """Cancela selección de límites."""
        self.result = None
        plt.close(self.figure)
