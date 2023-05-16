#! /usr/bin/env python3

"""
Full coupling (diffraction + radiation) for a triangular array,
comparison with the BEM code Capytaine by M. Ancellin
https://github.com/capytaine/capytaine
"""

import numpy as np
import floatcyl as fcl
import capytaine as cpt

r = 1.
draft = 1.
depth = 8.
beta = 0.
rho = 1000.
g = 9.81

m = rho * draft * np.pi * r**2
k = rho * g * np.pi * r**2

xb = np.array((0.,   2., 0.))
yb = np.array((-2.,  0., 3.))


cyl = cpt.VerticalCylinder(length=2*draft, radius=r,
                            nx=10, nr=10, ntheta=50)
cyl.keep_immersed_part()
cyl.add_translation_dof(name='Heave')
body_array = cyl.assemble_arbitrary_array(np.array((xb, yb)).T)

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
    M = m * np.eye(len(xb))
    K = k * np.eye(len(xb))

    Z = -omega**2*(M + A) - 1j*omega*B + K
    x_cap = np.linalg.solve(Z.data[0,:,:], f.data[0,0,:])

    x_cap_mat[ii, :] = x_cap

    print('oscillation amplitude, Capytaine', x_cap)
    print("phase = ", np.angle(x_cap)*180/np.pi, ' deg')

    ################## FLOATCYL SOLUTION ##################

    # 1) Dispersion relation
    wavenum = fcl.real_disp_rel(omega, depth)


    Nq = 33 #number of evanescent modes
    kq = fcl.imag_disp_rel(omega, depth, Nq)


    Nn = 4  #number of progressive modes


    # 2) isolated body geometry
    body = fcl.Cylinder(radius=r, draft=draft, omega=omega)


    # 3) define array
    cylArray = fcl.Array(beta=beta, depth=depth, k=wavenum, kq=kq, Nn=Nn, Nq=Nq,
                         omega=omega, water_density=rho, g=g)
    for jj in range(len(xb)):
        cylArray.add_body(xb[jj], yb[jj], body)

    # 4) compute isolated body behaviour (only once!)
    body.compute_diffraction_properties() #B
    body.compute_radiation_properties() #Btilde

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
np.save('x_fcl', x_fcl_mat)
np.save('x_cap', x_cap_mat)

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
