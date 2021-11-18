#! /usr/bin/env python3

"""
Demonstration of the meaning of matrix T_ij.
Here we investigate the transformation of the scattered wave
coefficients of body 0 into the incident wave coefficients of
body 1. One can try to change the spacing between bodies (relative
to their radius), the number of progressive and evanescent waves
and the scattered wave coefficients to see how the
approximation changes.
"""


import floatcyl as fcl
import numpy as np
import matplotlib.pyplot as plt


omega = 1.
depth = 10.
beta = 0.
rho = 1000.
g = 9.81

# Body positions
xc = [-10., 10.]
yc = [0., 0.]
# Radii of bodies (same)
a = 8
# Drafts of bodies (same)
draft = 0.5

# Real dispersion relation
k = fcl.real_disp_rel(omega, depth)
print('Wavelength = ', 2*np.pi/k, ' m')

Nq = 10 #number of evanescent modes
kq = fcl.imag_disp_rel(omega, depth, Nq)

Nn = 10 #number of progressive modes

# Isolated body geometry
body = fcl.Cylinder(radius=a, draft=draft)

# Define the array
cylArray = fcl.Array(beta=beta, depth=depth, k=k, kq=kq, Nn=Nn, Nq=Nq,
                     omega=omega, water_density=rho, g=g)
# Add the first body
cylArray.add_body(xc[0], yc[0], body)
# Add the second body
cylArray.add_body(xc[1], yc[1], body)

# Some properties of the objects...
print("dir(cylArray) = ", dir(cylArray))
print("dir(cylArray.bodies[0]) = ", dir(cylArray.bodies[0]))
print("cylArray.x = ", cylArray.x)
print("cylArray.y = ", cylArray.y)

# Define the vector of scattered coefficients
c_0_S = np.zeros((2*Nn+1)*(Nq+1))
from floatcyl.utils.utils import vector_index
# A single, unit, progressive wave from body 0
c_0_S[vector_index(0, 0, Nn, Nq)] = 1.
T_01 = cylArray.basis_transformation_matrix(0, 1, shutup=False)
c_1_I = T_01.T @ c_0_S

#print(c_0_S)
#print(c_1_I)



# Plots
# Define the grid
Lx = 50
nx = 50
Ly = 50
ny = 50
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
X, Y = np.meshgrid(x, y)


eta_0_S = np.zeros(X.flatten().shape)

# Loop on the wave coefficients
for ii in range(c_0_S.shape[0]):
    #Turn the single index of the coeff. vector to the double index
    #of progressive and evanescent waves (we already know that m=0)
    n, m = fcl.utils.utils.inverse_vector_indices(ii, Nn, Nq)
    psi_0_S = cylArray.scattered_basis_fs(n, m, k, a, xc[0], yc[0],
                                        X.flatten(), Y.flatten())
    eta_0_S = eta_0_S + psi_0_S*c_0_S[ii]

eta_0_S = eta_0_S.reshape(ny, nx)


eta_1_I = np.zeros(X.flatten().shape)

# Loop on the wave coefficients
for ii in range(c_1_I.shape[0]):
    #Turn the single index of the coeff. vector to the double index
    #of progressive and evanescent waves (we already know that m=0)
    n, m = fcl.utils.utils.inverse_vector_indices(ii, Nn, Nq)
    psi_1_I = cylArray.incident_basis_fs(n, m, k, depth, a, xc[1], yc[1],
                                        X.flatten(), Y.flatten())
    eta_1_I = eta_1_I + psi_1_I*c_1_I[ii]

eta_1_I = eta_1_I.reshape(ny, nx)


levels = np.linspace(0, 1.5, 10)

fig, ax = plt.subplots()
cs = ax.contourf(X, Y, np.abs(eta_0_S), levels=levels)
ax.set_title('eta_0_S')
#coolwarm is a diverging colormap, good for values centered around zero
cs.set_cmap('coolwarm')
cbar = plt.colorbar(cs)


fig, ax = plt.subplots()
cs = ax.contourf(X, Y, np.abs(eta_1_I), levels=levels)
ax.set_title('eta_1_I')
#coolwarm is a diverging colormap, good for values centered around zero
cs.set_cmap('coolwarm')
cbar = plt.colorbar(cs)


plt.show()
