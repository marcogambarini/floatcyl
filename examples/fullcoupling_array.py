#! /usr/bin/env python3


import floatcyl as fcl
import numpy as np

omega = 1.
depth = 30.
beta = 0.
rho = 1000.
g = 9.81
Nbodies = 100

a = 1.
spacing = 10.
draft = 0.5


# 1) Dispersion relation
k = fcl.real_disp_rel(omega, depth)

print('Wavelength = ', 2*np.pi/k, ' m')

Nq = 45 #number of evanescent modes
kq = fcl.imag_disp_rel(omega, depth, Nq)


Nn = 5  #number of progressive modes


# 2) isolated body geometry
body = fcl.Cylinder(radius=a, draft=draft)


# 3) define array
cylArray = fcl.Array(beta=beta, depth=depth, k=k, kq=kq, Nn=Nn, Nq=Nq,
                     omega=omega, water_density=rho, g=g)
for ii in range(Nbodies):
    cylArray.add_body(0., ii*spacing, body)

#compute isolated body behaviour
for body in cylArray.bodies:
    body.compute_diffraction_properties()
    body.compute_radiation_properties()

# build the full matrix and solve the problem
cylArray.solve()

print("rao = ", cylArray.rao)

# 4) compute the free surface elevation
# nx = 100
# ny = 40
# Lx = 50.
# Ly = 20.
# x = np.linspace(-Lx, Lx, nx)
# y = np.linspace(-Lx, Lx, ny)
# eta = cylArray.compute_free_surface(x, y)
