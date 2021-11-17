#! /usr/bin/env python3

"""
This example shows how an incident plane wave is projected on the
basis of incident waves. Increasing the number of progressive waves Nn,
the approximation becomes better.
Some of the inner workings of the code are exposed. 
"""


import floatcyl as fcl
import numpy as np
import matplotlib.pyplot as plt

omega = 2.
depth = 30.
beta = 0.
rho = 1000.
g = 9.81

# Body position
xc = 0.
yc = 0.
# Body radius
a = 1.

# Real dispersion relation
k = fcl.real_disp_rel(omega, depth)
print('Wavelength = ', 2*np.pi/k, ' m')

# No imaginary dispersion relation needed for incident plane waves!
Nq = 0

Nn = 20  #number of progressive modes


cylArray = fcl.Array(beta=beta, depth=depth, k=k, kq=[], Nn=Nn, Nq=Nq,
                     omega=omega, water_density=rho, g=g)
coeffs = cylArray.incident_wave_coeffs(k, beta, a, xc, yc, Nn, Nq)


# Define the grid
Lx = 55
nx = 50
Ly = 55
ny = 50
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
X, Y = np.meshgrid(x, y)

eta = np.zeros(X.flatten().shape)

# Loop on the wave coefficients
for ii in range(coeffs.shape[0]):
    #Turn the single index of the coeff. vector to the double index
    #of progressive and evanescent waves (we already know that m=0)
    n, m = fcl.utils.utils.inverse_vector_indices(ii, Nn, Nq)
    c = cylArray.incident_basis_fs(n, 0, k, depth, a, xc, yc,
                                        X.flatten(), Y.flatten())
    eta = eta + c*coeffs[ii]

eta = eta.reshape(ny, nx)


fig, ax = plt.subplots()
cs = ax.contourf(X, Y, np.real(eta))
ax.set_title('Real part ("snapshot")')
#coolwarm is a diverging colormap, good for values centered around zero
cs.set_cmap('coolwarm')
cbar = plt.colorbar(cs)

fig, ax = plt.subplots()
cs = ax.contourf(X, Y, np.abs(eta))
ax.set_title('Absolute value ("envelope")')
cbar = plt.colorbar(cs)

plt.show()
