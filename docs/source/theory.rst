Theory guide
============

The full theory can be found in Child's thesis (2011) and Yilmaz's paper (1998);
see the readme for references. In particular, there are direct references to the
former in the code documentation. Here, I will only show the big picture.

The aim is to solve the problem of computing forces on floating
bodies in wave fields, and their oscillation amplitude. The strategy
is to identify the way single bodies respond to incoming waves, in terms
of the forces they are subjected to and of the waves they re-emit
(diffraction and radiation). A considerable amount of time can be saved
with respect to other linear analysis methods, such as the BEM, if many
bodies are identical: their isolated behaviours will be the same, and will
only need to be computed once. The elevation of the free surface can
then be recovered as a post-processing stage.

The flow field is described by the Laplace equation. Since we are dealing
with cylinders, the equation is solved in cylindrical coordinates by
separation of variables. The boundary conditions imply that the vertical
dependence is in terms of hyperbolic functions, the azimuthal in terms of
harmonic functions, and the radial in terms of Bessel functions.
The solution can thus be expressed as a series. Each body will then
have a basis of incoming waves and a basis of outgoing waves.
It is important to choose how many
terms of the series to use: this should be done by analyzing convergence.
The parameters are the number of progressive modes (corresponding to
real solutions of the dispersion equation) and evanescent modes (corresponding
to imaginary solutions of the dispersion equation). Evanescent modes
become important when the bodies are close enough to each other.

First, we identify the way a fixed body responds to incident waves.
This is diffraction. The result is a matrix that connects the coefficients
of the incident wave expansion to the ones of the diffracted waves expansion.

Then, we study what waves are generated when a body oscillates with unit
amplitude. This is radiation. The result is a vector of coefficients for
outgoing waves.

Now, each body is subject to the incident wave field and to the diffracted
and radiated fields of all bodies. The problem is closed with a dynamic
equation that determines the oscillation amplitude of each body from the
forces caused by incident waves (Froude-Krylov force), diffracted waves
and radiated waves. If the body is connected to the bottom by a
viscoelastic element, this is where it has to be included.

At the moment, only heave (vertical) motions are implemented, as in
Child's thesis.
Surge and pitch could be added, following Yilmaz's article.
