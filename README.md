<p align="center">
  <br />
  <br />
  <a href="https://www.iaa.es/">
    <img
      src="https://www.iaa.csic.es/sites/default/files/banners/news/banner_web_iaa_57.png"
      alt="ViewCube">
  </a>
</p>

<!-- Badges -->
<p align="center"> 
   <!-- Latest Viewcube version updated to PyPI -->
  <a href="https://pypi.org/project/ViewCube/">
      <img src="https://badgen.net/pypi/v/viewcube">
  </a>

   <!-- Python version required -->
  <a href="https://pypi.org/project/ViewCube/">
      <img src="https://badgen.net/pypi/python/viewcube">
  </a>

   <!-- PyPI license -->
  <a href="https://pypi.org/project/ViewCube/">
      <img src="https://badgen.net/pypi/license/viewcube">
  </a>

  <!-- Code style -->
  <a href="https://styles.goatbytes.io/lang/python/">
      <img src="https://img.shields.io/badge/Style%20Guide-Python-3776AB.svg?style=flat&labelColor=black&color=3776AB&logo=python">
  </a>
</p>

<br />
<p align="center">
  <a href="#documentation"><b>Documentation</b></a> •
  <a href="#installation"><b>Installation</b></a> •
  <a href="#usage"><b>Usage</b></a> •
  <a href="https://pypi.org/project/ViewCube/"><b>PyPI Repository</b></a>
</p>
<br />

---

# ViewCube - Datacube Visualization [& Sonification] Tool

**ViewCube** is a user-friendly datacube visualizer designed to streamline the exploration and 
analysis of multi-dimensional datasets. It enables easy visualization and interaction with 
datacubes, while its **Sonicube** interface offers auditory exploration through **sonification**, 
enriching the data analysis process.

Additionally, the tool allows for the exploration of raw-stacked spectra (RSS) via its **ViewRSS** 
interface, provided a fiber position table is available.

## Documentation

For detailed information on ViewCube, visit the official documentation:

[https://viewcube.readthedocs.io](https://viewcube.readthedocs.io/)

## Installation

We recommend referring to the ViewCube documentation for comprehensive guidance:

[https://viewcube.readthedocs.io](https://viewcube.readthedocs.io/)

To get started quickly, just run:
```
pip install viewcube
```

For more information, visit the [ViewCube PyPI page](https://pypi.org/project/ViewCube/).

## Instalación local y uso de ViewCube

En la carpeta donde quieras trabajar, crea un entorno virtual para Python con la siguiente instrucción:
```bash
python -m venv proyecto
```

Navegue y active el entorno virtual con las siguientes instrucciones:
```bash
cd proyecto
./activate
```

Ahora, clone el repositorio dentro del entorno virtual, tal que así:
```bash
git clone https://github.com/AnaJaneMM/viewcube.git
```
> Tanto la rama `main` como `develop` contienen el código funcional.

Instala el programa mediante
```bash
cd proyecto
cd viewcube
pip install .
```

Esto instalará todas las dependencias que se han establecido en el fichero `pyproject.toml`.

Abre, desde tu IDE preferido (aunque recomendamos PyCharm) la carpeta `proyecto/viewcube`. Pulsa en el botón verde de 'Ejecutar' y se abrirá el entorno gráfico.
![gui de ejemplo](/gui.png)

Para abrir un fichero FITS, pulsa a la izquierda en 'Fichero FITS principal' y selecciona uno desde el explorador. Ahora, abajo del todo pulsa en el botón "Cargar". Aparecerá una ventana con la información del ficheor y luego, se mostrarán ambas gráficas en pantalla.

## Documentation

For detailed instructions and information, consult the ViewCube documentation:

[https://viewcube.readthedocs.io](https://viewcube.readthedocs.io/)

## Usage

To visualize a datacube, follow these steps:

1. Open a terminal window.
2. Navigate to the folder where your datacube file (`name_cube.fits`) is located.
3. Enter the following command:
    ```
    ViewCube name_cube.fits
    ```

The first time you run ``ViewCube`` you may need to create a configuration file:
```
ViewCube --config-file
```


## Cheat Sheet

Refer to the ``ViewCube.pdf`` cheat sheet PDF file in the GitHub latex directory or 
the [documentation](https://viewcube.readthedocs.io/) for a list of keyboard 
shortcuts and quick operations to enhance your experience.

Start exploring your datacubes with ViewCube and gain valuable insights effortlessly. For any further assistance or feedback, please refer to the documentation or reach out to our support team. Happy visualizing!

## How to cite this package

If you use this package in your research—whether for visualization, sonification, or both—please 
cite it by including as much of the following information as possible:

Adrián García Riber, Rubén García-Benito, Francisco Serradilla. *Interactive multimodal integral 
field spectroscopy*, RAS Techniques and Instruments, Volume 3, Issue 1, January 2024, Pages 748-758
[https://doi.org/10.1093/rasti/rzae049](https://doi.org/10.1093/rasti/rzae049)
