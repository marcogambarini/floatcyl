#! /usr/bin/env python3


import floatcyl as fcl
import numpy as np

omega = 3.
depth = 30.
beta = 0.
rho = 1000.
g = 9.81
Nbodies = 4

a = 1.
spacing = 10.
draft = 0.5


# 1) Dispersion relation
k = fcl.real_disp_rel(omega, depth)

print('Wavelength = ', 2*np.pi/k, ' m')

Nq = 50 #number of evanescent modes
kq = fcl.imag_disp_rel(omega, depth, Nq)


Nn = 5 #number of progressive modes


# 2) isolated body behaviour
body = fcl.Cylinder(radius=a, draft=draft, depth=depth, k=k,
                    kq=kq, Nn=Nn, Nq=Nq, omega=omega)

body.compute_diffraction_properties()
body.compute_radiation_properties()


# 3) define array
cylArray = fcl.Array()
for ii in range(Nbodies):
    cylArray.add_body(0., ii*spacing, body)
