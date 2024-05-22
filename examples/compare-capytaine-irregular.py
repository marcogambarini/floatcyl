#! /usr/bin/env python3

"""
Full coupling for a WEC array, with controls and irregular waves
comparison with the BEM code Capytaine by M. Ancellin
https://github.com/capytaine/capytaine
"""

import numpy as np
import floatcyl as fcl
import capytaine as cpt
from floatcyl.utils.utils import discrete_PM_spectrum

r = 2.
draft = 0.5
depth = 50.
beta = 0.
rho = 1000.
g = 9.81

Tp = 5.83
Te = Tp*0.86
Hs = 1.53
Nbins = 30
Nq = 45
Nn = 5

m = rho * draft * np.pi * r**2
k = rho * g * np.pi * r**2

# Device positions
n1 = 8 # bodies on upwave arc
n2 = 7 # bodies on downwave arc
ntot = n1 + n2
xc = -20.
R1 = 40
R2 = 50
R = np.zeros(ntot)
theta = np.zeros(ntot)
R[:n1] = R1
R[n1:] = R2
theta[:n1] = np.linspace(-60, 60, n1)*np.pi/180
theta[n1:] = np.linspace(-45, 45, n2)*np.pi/180
xb = R*np.cos(theta) + xc
yb = R*np.sin(theta)

# Device controls
damp_vec = np.array([37554.38422118,  38682.95944343,  36520.89104467,  39450.83540906,
  39475.53078078,  36526.68124425,  38687.71993094,  37565.12439647,
  32739.78930619,  29573.95704838,  28476.04606604,  30227.11800111,
  28478.43330329,  29542.23811182,  32753.35363566])
stiff_vec = np.array([-22446.1261122, -23124.71426704, -28484.59998492, -25594.87315308,
 -25650.22497196, -28431.93827236, -23146.01308952, -22502.64856293,
 -26188.88926718, -30097.81235593, -35721.81791964, -35574.80944299,
 -35706.24515876, -30068.97704285, -26228.40236788])

cyl = cpt.VerticalCylinder(length=2*draft, radius=r,
                            nx=10, nr=10, ntheta=50)
cyl.keep_immersed_part()
cyl.add_translation_dof(name='Heave')
body_array = cyl.assemble_arbitrary_array(np.array((xb, yb)).T)

Nbodies = len(xb)
omegavec, ampvec = discrete_PM_spectrum(Te, Hs, Nbins, crit='F')

P_cap_mat = np.zeros((Nbins, Nbodies))
P_fcl_mat = np.zeros((Nbins, Nbodies))


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

    Z = -omega**2*(M + A) - 1j*omega*(B + np.diag(damp_vec)) + K + np.diag(stiff_vec)
    x_cap = np.linalg.solve(Z.data[0,:,:], f.data[0,0,:])

    print('oscillation amplitude, Capytaine', x_cap)
    print("phase = ", np.angle(x_cap)*180/np.pi, ' deg')
    print('powers = ')
    for jj in range(Nbodies):
        P_cap_mat[ii, jj] = 0.5*omega**2*ampvec[ii]**2*damp_vec[jj]*np.abs(x_cap[jj])**2
        print(P_cap_mat[ii, jj])

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
    cylArray.update_controls(damp_vec, stiff_vec)

    # 4) compute isolated body behaviour (only once!)
    body.compute_diffraction_properties() #B
    body.compute_radiation_properties() #Btilde

    # 5) build the full matrix and solve the problem
    cylArray.solve()

    x_fcl = np.array(cylArray.rao)[:, 0]

    print("oscillation amplitude, Floatcyl = ", x_fcl)
    print("phase = ", np.angle(x_fcl)*180/np.pi, ' deg')
    print("powers = ")
    for jj in range(Nbodies):
        P_fcl_mat[ii, jj] = 0.5*omega**2*ampvec[ii]**2*damp_vec[jj]*np.abs(x_fcl[jj])**2
        print(P_fcl_mat[ii, jj])

    ######################## COMPARISON #######################
    amp_diff = (np.abs(x_fcl) - np.abs(x_cap))/(np.abs(x_cap))
    print('amplitude difference = ', amp_diff*100, ' %')
    phase_diff = (np.angle(x_fcl) - np.angle(x_cap)) * 180/np.pi
    print('phase difference = ', phase_diff, 'deg')

np.savez('compare-cpt-irregular.npz', P_cpt_mat=P_cpt_mat, P_fcl_mat=P_fcl_mat)
