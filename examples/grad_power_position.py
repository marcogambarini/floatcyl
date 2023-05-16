import floatcyl as fcl
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

omega = 1.
H = 1.
depth = 10.
beta = 0
rho = 1000.
g = 9.81
a = 1.
draft = 0.5

h = 0.000001

denseops = False

# Dispersion relation
k = fcl.real_disp_rel(omega, depth)

print('Wavelength = ', 2*np.pi/k, ' m')

Nq = 8 #number of evanescent modes
kq = fcl.imag_disp_rel(omega, depth, Nq)

Nn = 3  #number of progressive modes

# isolated body geometry
body = fcl.Cylinder(radius=a, draft=draft, omega=omega, depth=depth,
                    gamma=50000, delta=4000)


# define array
cylArray0 = fcl.Array(beta=beta, depth=depth, k=k, kq=kq, Nn=Nn, Nq=Nq,
                    omega=omega, water_density=rho, g=g, denseops=denseops)

var_index = 2

###################COMPUTE MATRICES
x_0 = np.array((0., 3., 6., 9., 0., 3., 6., 9., 0., 3., 6., 9.))
y_0 = np.array((0., 0., 0., 0., 3., 3., 3., 3., 6., 6., 6., 6.))

Nbodies = len(x_0)

for ii in range(Nbodies):
    cylArray0.add_body(x_0[ii], y_0[ii], body)

body.compute_diffraction_properties()
body.compute_radiation_properties()

cylArray0.solve()
rao = cylArray0.rao

P_array0 = 0
for ii in range(Nbodies):
    P_array0 = P_array0 + np.abs(rao[ii])**2 * omega*omega /2 * H*H * body.gamma
J_0 = -P_array0

print('Initial cost = ', J_0)

start_time = perf_counter()
cylArray0.adjoint_equations()
gradJ = cylArray0.gradientJ() # separated gradients in x and y
print('time for gradient computation through adjoint = ', perf_counter() - start_time)

def compute_fd(x_h, y_h):
    # define array
    cylArray_h = fcl.Array(beta=beta, depth=depth, k=k, kq=kq, Nn=Nn, Nq=Nq,
                        omega=omega, water_density=rho, g=g, denseops=denseops)
    for ii in range(Nbodies):
        cylArray_h.add_body(x_h[ii], y_h[ii], body)

    cylArray_h.solve()
    rao = cylArray_h.rao

    P_array_h = 0
    for ii in range(Nbodies):
        P_array_h = P_array_h + np.abs(rao[ii])**2 * omega*omega /2 * H*H * body.gamma
    J_h = -P_array_h

    fd = (J_h - J_0)/h
    return fd

fd_x = np.zeros(Nbodies)
fd_y = np.zeros(Nbodies)

start_time = perf_counter()
for kk in range(Nbodies):
    x_h = x_0.copy()
    y_h = y_0.copy()
    x_h[kk] += h

    fd_x[kk] = compute_fd(x_h, y_h)

for kk in range(Nbodies):
    x_h = x_0.copy()
    y_h = y_0.copy()
    y_h[kk] += h

    fd_y[kk] = compute_fd(x_h, y_h)
print('time for gradient computation by fd = ', perf_counter() - start_time)

gradJ_merged = np.block([gradJ[0], gradJ[1]])
fd_merged = np.block([fd_x, fd_y])

print('gradient = ', gradJ_merged)
print('finite differences = ', fd_merged)
print('relative error = ', np.linalg.norm(fd_merged - gradJ_merged)/np.linalg.norm(gradJ_merged))

# The exact value of divergence zero:
# displacing all bodies by the same vector must not change the power
print('divergence = ', np.sum(gradJ_merged))
