#! /usr/bin/env python3

"""
Full coupling (diffraction + radiation) for a triangular array
of bodies of different sizes,
comparison with the BEM code Capytaine by M. Ancellin
https://github.com/capytaine/capytaine
"""

import numpy as np
import floatcyl as fcl
import capytaine as cpt

depth = 10.
beta = 0.
rho = 1000.
g = 9.81

r = np.array((2., 3., 3.))
draft = np.array((1., 2., 2.))
m = rho * draft * np.pi * r**2
k = rho * g * np.pi * r**2

xb = np.array((0.,  8., 12.))
yb = np.array((0., -4.,  2.))

Nbodies = len(r)

for ii in range(Nbodies):
    cyl = cpt.VerticalCylinder(length=2*draft[ii], radius=r[ii],
                               center=(xb[ii], yb[ii], 0),
                               nx=10, nr=10, ntheta=50)
    cyl.keep_immersed_part()
    cyl.add_translation_dof(name='Heave')
    if ii==0:
        body_array = cyl
    else:
        body_array = body_array + cyl

#body_array.show()
Ntest = 50
Nbodies = len(xb)
omegavec = np.linspace(0.1, 5, Ntest)
np.save('omegavec', omegavec)


x_cap_mat = np.zeros((Ntest, Nbodies), dtype=complex)
x_fcl_mat = np.zeros((Ntest, Nbodies), dtype=complex)



for ii, omega in enumerate(omegavec):


    print('\n--------------------------------\nii=', ii,', omega = ', omega)

    problems = [cpt.RadiationProblem(body=body_array, radiating_dof=dof,
                            omega=omega, sea_bottom=-depth) for dof in body_array.dofs]
    problems += [cpt.DiffractionProblem(body=body_array, wave_direction=beta,
                            omega=omega, sea_bottom=-depth)]

    # Solves the problem
    solver = cpt.BEMSolver()
    results = solver.solve_all(problems)
    data = cpt.assemble_dataset(results)

    A = data['added_mass']
    B = data['radiation_damping']
    f = data['diffraction_force'] + data['Froude_Krylov_force']
    M = np.diag(m)
    K = np.diag(k)

    Z = -omega**2*(M + A) - 1j*omega*B + K
    x_cap = np.linalg.solve(Z.data[0,:,:], f.data[0,0,:])

    x_cap_mat[ii, :] = x_cap

    print('oscillation amplitude, Capytaine', x_cap)
    print("phase = ", np.angle(x_cap)*180/np.pi, ' deg')

    ################## FLOATCYL SOLUTION ##################

    # Dispersion relation
    wavenum = fcl.real_disp_rel(omega, depth)


    Nq = 33 #number of evanescent modes
    kq = fcl.imag_disp_rel(omega, depth, Nq)


    Nn = 12  #number of progressive modes


    # define array
    cylArray = fcl.Array(beta=beta, depth=depth, k=wavenum, kq=kq, Nn=Nn, Nq=Nq,
                         omega=omega, water_density=rho, g=g)
    body_list = []
    for jj in range(Nbodies):
        # isolated body geometry
        body_list.append(fcl.Cylinder(radius=r[jj], draft=draft[jj], omega=omega))
        cylArray.add_body(xb[jj], yb[jj], body_list[jj])
        # compute isolated body behaviour (all bodies are different)
        body_list[jj].compute_diffraction_properties()
        body_list[jj].compute_radiation_properties()

    # 5) build the full matrix and solve the problem
    cylArray.solve()

    x_fcl = np.array(cylArray.rao)[:, 0]
    #x_fcl = np.array(cylArray.rao)
    x_fcl_mat[ii, :] = x_fcl

    print("oscillation amplitude, Floatcyl = ", x_fcl)
    print("phase = ", np.angle(x_fcl)*180/np.pi, ' deg')

    ######################## COMPARISON #######################
    amp_diff = (np.abs(x_fcl) - np.abs(x_cap))/(np.abs(x_cap))
    print('amplitude difference = ', amp_diff*100, ' %')
    phase_diff = (np.angle(x_fcl) - np.angle(x_cap)) * 180/np.pi
    print('phase difference = ', phase_diff, 'deg')

# optional: save the results for future plotting
np.save('x_fcl_varsize', x_fcl_mat)
np.save('x_cap_varsize', x_cap_mat)

######################### PLOTS ########################
import matplotlib.pyplot as plt

body_index = 1 # index of the body for which to plot the results

# Correct and unwrap phases for comparison
fcl_phase = np.unwrap(np.angle(x_fcl_mat[:, body_index])-np.pi/2)*180/np.pi
cap_phase = np.unwrap(np.angle(x_cap_mat[:, body_index]))*180/np.pi

fig, ax = plt.subplots(2, 1)
ax[0].plot(omegavec, np.abs(x_fcl_mat[:, body_index]), 'o', label='Fcl')
ax[0].plot(omegavec, np.abs(x_cap_mat[:, body_index]), '-', label='Cap')
ax[1].plot(omegavec, fcl_phase, 'o', label='Fcl')
ax[1].plot(omegavec, cap_phase, '-', label='Cap')
ax[1].legend()
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel('Amplitude')
ax[1].set_ylabel('Phase (deg)')

plt.show()
