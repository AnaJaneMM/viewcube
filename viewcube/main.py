# main.py - Punto de entrada completo
"""Punto de entrada principal para ViewCube refactorizado."""

import argparse
import os
import sys
import matplotlib
import matplotlib.pyplot as plt

from config.configuration_manager import ConfigurationManager
from viewcube.ui.controllers import MainController
from ui.viewers.rss_viewer import RSSViewer

VERSION = "1.0.0"


def parse_arguments() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ViewCube: Visualizador de datos astronómicos"
    )

    # Argumentos principales
    parser.add_argument("name", type=str, nargs="*", help="Archivo FITS")
    parser.add_argument("-c", "--comparison", type=str, help="Archivo de comparación")
    parser.add_argument("-p", "--position", type=str, help="Tabla de posiciones")

    # Configuración
    parser.add_argument("--config-file", action="store_true", help="Crear archivo de configuración")
    parser.add_argument("--config", type=str, help="Archivo de configuración personalizado")
    parser.add_argument("-v", "--version", action="store_true", help="Mostrar versión")

    # Parámetros de datos
    parser.add_argument("--data", type=int, help="Extensión de datos")
    parser.add_argument("--error", type=int, help="Extensión de errores")
    parser.add_argument("--flag", type=int, help="Extensión de flags")
    parser.add_argument("--header", type=int, default=0, help="Extensión de cabecera")
    parser.add_argument("--wave", help="Extensión de longitud de onda")
    parser.add_argument("-s", "--specaxis", type=int, help="Dimensión espectral")

    # Configuración visual
    parser.add_argument("-b", "--backend", type=str, help="Backend de matplotlib")
    parser.add_argument("-y", "--style", type=str, help="Estilo de matplotlib")
    parser.add_argument("--filter-dir", type=str, default="filters/", help="Directorio de filtros")
    parser.add_argument("--default-filter", type=str, help="Filtro por defecto")
    parser.add_argument("--norm", type=str, default="sqrt", help="Función de normalización")

    # Factores multiplicativos
    parser.add_argument("--fo", type=float, default=1.0, help="Factor para archivo original")
    parser.add_argument("--fc", type=float, default=1.0, help="Factor para comparación")

    # Opciones adicionales
    parser.add_argument("-i", "--ivar", action="store_true", help="Conversión IVAR a error")
    parser.add_argument("-a", "--angle", type=float, help="Ángulo de rotación")
    parser.add_argument("-e", "--extension", action="store_true", help="Tabla en extensión")

    return parser.parse_args()


def main() -> None:
    """Función principal."""
    args = parse_arguments()

    # Manejar opciones especiales
    if args.version:
        print(f'ViewCube version: {VERSION}')
        sys.exit()

    # Configurar matplotlib
    if args.backend:
        matplotlib.use(args.backend)
    if args.style:
        plt.style.use(args.style.split(','))

    # Gestión de configuración
    config_manager = ConfigurationManager(args.config)

    if args.config_file:
        config_manager.create_config_file(force=True)
        sys.exit()

    # Validar argumentos
    if not args.name:
        print("Error: Se requiere archivo FITS")
        sys.exit(1)

    filename = args.name[0]
    if not os.path.exists(filename):
        print(f"Error: Archivo '{filename}' no encontrado")
        sys.exit(1)

    # Cargar configuración
    config = config_manager.load_config()

    # Actualizar con argumentos de línea de comandos
    cli_args = {
        'fitscom': args.comparison,
        'ptable': args.position,
        'dfilter': args.filter_dir,
        'default_filter': args.default_filter,
        'norm': args.norm,
        'fo': args.fo,
        'fc': args.fc,
        'exdata': args.data,
        'exerror': args.error,
        'exflag': args.flag,
        'exhdr': args.header,
        'exwave': args.wave,
        'specaxis': args.specaxis,
        'ivar': args.ivar,
        'angle': args.angle,
        'extension': args.extension
    }

    # Filtrar valores None
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value

    # Crear controlador apropiado
    try:
        if config.get('ptable') or config.get('extension'):
            # Usar visor RSS
            viewer = RSSViewer(filename, config.get('ptable'), **config)
        else:
            # Usar controlador principal
            viewer = MainController(filename, **config)

        # Mostrar información de controles
        print("\n" + "=" * 60)
        print("VIEWCUBE - CONTROLES:")
        print("=" * 60)
        print("Navegación:")
        print("  - Click izquierdo: Seleccionar spaxel/fibra")
        print("  - Click derecho: Mostrar espectro")
        print("  - Movimiento del mouse: Vista previa")
        print("\nFiltros:")
        print("  - 't': Siguiente filtro")
        print("  - 'T': Filtro anterior")
        print("\nVisualización:")
        print("  - '*': Limpiar selecciones")
        print("  - 's': Guardar espectros")
        print("  - 'w': Gestor de ventanas")
        print("  - 'l': Límites de longitud de onda")
        print("  - 'Y': Límites de flujo")
        print("  - 'q': Salir")
        print("=" * 60)

        # Ejecutar
        viewer.run()

    except Exception as e:
        print(f"Error al inicializar ViewCube: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
