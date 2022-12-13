User guide
==========

The best way to understand how to use the code is to read
the examples on github. This page highlights the main steps
of a typical computation.

0. Import floatcyl.
1. Solve the real and imaginary dispersion relation. This is the
   right time to define the number of progressive and
   evanescent modes that you want to consider (you should then check
   convergence relative to these parameters).
2. Define the geometry of the bodies.
3. Define the array and the problem parameters.
4. Compute the diffraction  and radiation matrices for
   each isolated body. If there are groups of identical bodies, this
   needs to be done only once per group.
   *This is the most computationally expensive part, and computing
   the isolated body properties only once is the very reason why
   the method is powerful for arrays.*
5. Solve the problem to find RAOs.
6. (optional) Compute the free surface elevation (complex field).
   If you want the wave envelope at each point, take the absolute
   value. If you want a snapshot, take the real part (or the
   imaginary part, or a combination). If you want an animation,
   multiply it by exp(-j omega t).


Examples
--------

The examples are in order of complexity.

1. inc_field.py: decomposition of an incident plane wave
   in the basis of incident cylindrical waves.
2. basis_transf.py: meaning of matrix T_ij.
3. fullcoupling_array.py: simulation of a line array of cylinders.
4. compare_capytaine.py: simulation of a triangular array of cylinders
   and validation against open-source BEM solver Capytaine (needs
   a working installation of Capytaine)
