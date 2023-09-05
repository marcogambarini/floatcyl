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

h = 1

# Dispersion relation
k = fcl.real_disp_rel(omega, depth)

print('Wavelength = ', 2*np.pi/k, ' m')

Nq = 8 #number of evanescent modes
kq = fcl.imag_disp_rel(omega, depth, Nq)

Nn = 3  #number of progressive modes

damping = 50000
stiffness = 4000

# isolated body geometry
body = fcl.Cylinder(radius=a, draft=draft, omega=omega, depth=depth,
                    gamma=damping, delta=stiffness)


# define array
cylArray0 = fcl.Array(beta=beta, depth=depth, k=k, kq=kq, Nn=Nn, Nq=Nq,
                    omega=omega, water_density=rho, g=g, H=H)

var_index = 2

###################COMPUTE MATRICES
x_0 = np.array((0., 3., 6., 9., 0., 3., 6., 9., 0., 3., 6., 9.))
y_0 = np.array((0., 0., 0., 0., 3., 3., 3., 3., 6., 6., 6., 6.))

Nbodies = len(x_0)

for ii in range(Nbodies):
    cylArray0.add_body(x_0[ii], y_0[ii], body)

cylArray0.solve()
J_0 = -cylArray0.compute_power()

print('J_0 = ', J_0)

start_time = perf_counter()
cylArray0.adjoint_equations()
_, gradJ = cylArray0.gradientJ_dampstiff()
print('time for gradient computation through adjoint = ', perf_counter() - start_time)

def compute_fd(c):
    # update array
    cylArray0.update_controls(damping*np.ones(Nbodies), c)

    cylArray0.solve()
    J_h = -cylArray0.compute_power()


    fd = (J_h - J_0)/h
    return fd

fd_gamma = np.zeros(Nbodies)

start_time = perf_counter()
for kk in range(Nbodies):
    c = stiffness * np.ones(Nbodies)
    c[kk] += h
    fd_gamma[kk] = compute_fd(c)


print('time for gradient computation by fd = ', perf_counter() - start_time)

print('gradient = ', gradJ)
print('finite differences = ', fd_gamma)
print('relative error = ', np.linalg.norm(gradJ - fd_gamma)/np.linalg.norm(gradJ))
