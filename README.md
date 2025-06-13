# floatcyl
Implementation of wave interaction theory for floating cylinders and of gradient flow optimization tools.

The code is based on the works

[1] O. Yilmaz, *Hydrodynamic interactions of waves with group of truncated vertical cylinders*, Journal of waterway, port, coastal, and ocean engineering, 1998

[2] B. F. M. Child, *On the configuration of arrays of floating wave energy converters*, PhD Thesis, University of Edinburgh, 2011

[3] J. Gallizioli, *Optimization of WEC arrays: A new approach for the combined optimization of positions and damping coefficients*, MSc Thesis, Politecnico di Milano, 2022

[4] M. Gambarini, G. Ciaramella, E. Miglio, *A gradient flow approach for combined layout-control design of wave energy parks*, preprint, 2024, <https://arxiv.org/abs/2409.10200>

To install, enter the directory and use the command
`pip3 install -e .`
This is installation for developers: if you update the code, you will immediately use the updated version when you load the module in Python.

To see the documentation, enter folder docs and run
`make html`.
A file index.html will appear in docs/build.
This requires `sphinx`, `sphinx_rtd_theme` and `myst_parser`, which can be installed with `pip`.
An already built documentation is available [here](https://marcogambarini.github.io/).

Importing floatcyl automatically imports the hydrodynamic solver, together with the functions for computing gradients, but not the gradient flow submodule.
The latter can be imported as `import floatcyl.gradflow` and it requires the installation of Firedrake and gmsh:
- It is recommended to install Firedrake inside a virtual environment, as indicated in <https://www.firedrakeproject.org/download.html>.
- gmsh for Python can be installed following <https://pypi.org/project/gmsh/>.
Before installing floatcyl and every time the gradflow tools of floatcyl are used, activate the Firedrake virtual environment. 
It is convenient to create an alias for activating the Firedrake virtual environment: this can be done by adding `alias sourceFiredrake='source ~/firedrake/bin/activate'` to your .bashrc file.

The examples in the gradflow folder allow reproducing the results of [4]. Subdirectories square and cutsquare contain `run_tests.sh`, which
uses the parameters of configuration (.ini) files. Plots and tables can be produced using `run_plots.sh` and `make-table.py`. Tests were run with Python 3.10.12.

