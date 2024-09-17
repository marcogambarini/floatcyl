# floatcyl
Implementation of wave interaction theory for floating cylinders and of gradient flow optimization tools.

The code is based on the works
[1] O. Yilmaz, *Hydrodynamic interactions of waves with group of truncated vertical cylinders*, Journal of waterway, port, coastal, and ocean engineering, 1998
[2] B. F. M. Child, *On the configuration of arrays of floating wave energy converters*, PhD Thesis, University of Edinburgh, 2011
[3] J. Gallizioli, *Optimization of WEC arrays: A new approach for the combined optimization of positions and damping coefficients*, MSc Thesis, Politecnico di Milano, 2022
[4] M. Gambarini, G. Ciaramella, E. Miglio, *A gradient flow approach for combined layout-control design of wave energy parks*, to appear, 2024

To install, enter the directory and use the command
`pip3 install -e .`
This is installation for developers: if you update the code, you will immediately use the updated version when you load the module in Python.

To see the documentation, enter folder docs and run
`make html`.
A file index.html will appear in docs/build.

Importing floatcyl automatically imports the hydrodynamic solver, together with the functions for computing gradients, but not the gradient flow submodule.
The latter can be imported as `import floatcyl.gradflow` and it requires the installation of Firedrake and gmsh.
It is recommended to install Firedrake inside a virtual environment, as indicated in https://www.firedrakeproject.org/download.html.
Before installing floatcyl and everytime the gradflow tools of floatcyl are used, activate the Firedrake virtual environment.

Checking out the gradflow branch allows using optimization routines. The examples in the gradflow folder allow reproducing the results of [4].
