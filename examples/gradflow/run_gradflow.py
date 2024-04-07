#! /usr/bin/env python3

import numpy as np
import configparser
import csv
import floatcyl as fcl
from floatcyl.gradflow.random_pos import random_in_square
from floatcyl.gradflow.constr_jac import PolyConstraint
from floatcyl.gradflow.flowmap import Flowmap
from floatcyl.gradflow.rkflow import *
import sys

#################### READ PARAMETER FILE ####################

# Get parameter file name as command line argument or set to default
if len(sys.argv) > 0:
    paramFileName = sys.argv[1]
else:
    paramFileName = 'params.ini'

class StrictConfigParser(configparser.RawConfigParser):
    # config parser which raises exception instead of having fallbacks
    def get(self, section, option, raw=False, vars=None, fallback=None):
        print('reading section ', section, ', option ', option)
        var = super().get(section, option, raw=raw, vars=vars, fallback=None)
        if var is None:
            raise RuntimeError("No option %r in section: %r" %
                       (option, section))
        else:
            return var


conf = StrictConfigParser()
conf.read(paramFileName)

run_name = conf['rundata'].get('run_name')

rho = conf['environment'].getfloat('rho')
g = conf['environment'].getfloat('g')
depth = conf['environment'].getfloat('depth')

R = conf['devices'].getfloat('R')
draft = conf['devices'].getfloat('draft')

alpha_slam = conf['slamming'].getfloat('alpha_slam')

gen_damping = conf['control'].getfloat('gen_damping')
gen_stiffness = conf['control'].getfloat('gen_stiffness')

Nbodies = conf['array'].getint('Nbodies')
min_dist = conf['array'].getfloat('min_dist')
vertex_file = conf['array'].get('vertex_file')
seed = conf['array'].getint('seed')
rect_factor = conf['array'].getfloat('rect_factor')
h_outer = conf['array'].getfloat('h_outer')
h_inner = conf['array'].getfloat('h_inner')
save_pvd = conf['array'].getboolean('save_pvd')
projmethod = conf['array'].get('projmethod')
nu = conf['array'].getfloat('nu')

spectrum_from_file = conf['waves'].getboolean('spectrum_from_file')
if spectrum_from_file:
    spectrum_file = conf['waves'].get('spectrum_file')
    wave_data = np.load(spectrum_file)
    Hvec = 2*wave_data['amplitude'] # significant wave heights
    omegavec = wave_data['omega']
    beta = 0.
    Nf = len(omegavec) # number of frequencies (spectral components)
else:
    Te = conf['waves'].getfloat('Te')
    Hs = conf['waves'].getfloat('Hs')
    thres = conf['waves'].getfloat('thres')
    Nf = conf['waves'].getint('Nf')
    crit = conf['waves'].get('crit')
    omegavec, amplitude = fcl.utils.utils.discrete_PM_spectrum(
                                Te, Hs, Nf, filename=None, thres=thres,
                                plot=False, crit=crit)
    Hvec = 2*amplitude
beta = conf['waves'].getfloat('beta') # in degrees

Nn = conf['numerics'].getint('Nn')
Nq = conf['numerics'].getint('Nq')
dt_0 = conf['numerics'].getfloat('dt_0')
stop_tol = conf['numerics'].getfloat('stop_tol')
Tfin = conf['numerics'].getfloat('Tfin')
method = conf['numerics'].get('method')
adapt_CG_tol = conf['numerics'].getboolean('adapt_CG_tol')
if adapt_CG_tol:
    CG_tol = None
else:
    CG_tol = conf['numerics'].getfloat('CG_tol', None)
if method=='RK12':
    adapt_RK_tol = conf['numerics'].getboolean('adapt_RK_tol')
    if adapt_RK_tol:
        RK_atol = None
        RK_rtol = None
    else:
        RK_atol = conf['numerics'].getfloat('RK_atol')
        RK_rtol = conf['numerics'].getfloat('RK_rtol')
monitor = conf['numerics'].getboolean('monitor')

######################### DOMAIN SETUP #########################
# Read vertex file
vertices = []
with open(vertex_file, newline='') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        vertices.append(np.array(row, dtype=float))
vertices = np.array(vertices)

# Random initial guess for the positions: devices are placed in a
# square containing the prescribed domain
x0, y0 = random_in_square(min_dist/2,
        np.min(vertices[:,0]), np.max(vertices[:,0]),
        np.min(vertices[:,1]), np.max(vertices[:,1]),
        Nbodies, seed=seed)


print('\nCalling gmsh to build the grid...')
# Initialize domain constraints
domain_constr = PolyConstraint(vertices, meshFileName=run_name,
                 rect_factor=rect_factor, h_outer=h_outer,
                 h_inner=h_inner, save_pvd=save_pvd,
                 projmethod=projmethod, nu=nu)
print('Done')

####################### SPECTRUM PRINTOUT ######################
print('omegavec = ', omegavec)
print('Hvec = ', Hvec)


######################### SOLVER SETUP #########################
wavenum = []
cylArrays = []
for ii in range(Nf):
    print('\nFrequency ', ii+1, '/', Nf)
    wavenum.append(fcl.real_disp_rel(omegavec[ii], depth)) # real wavenumber
    kq = fcl.imag_disp_rel(omegavec[ii], depth, Nq) # imaginary wavenumbers

    # define a single device
    device = fcl.Cylinder(radius=R, draft=draft, depth=depth,
                k=wavenum[ii], kq=kq.copy(), Nn=Nn, Nq=Nq, omega=omegavec[ii],
                water_density=rho, g=g, gamma=gen_damping, delta=gen_stiffness)
    # compute isolated body behaviour (only once!)
    device.compute_diffraction_properties()
    device.compute_radiation_properties()

    print('Finished computing single-device properties')

    # Array initialization
    cylArray = fcl.Array(beta=beta*np.pi/180, depth=depth,
                         k=wavenum[ii], kq=kq.copy(),
                         Nn=Nn, Nq=Nq, omega=omegavec[ii], water_density=rho, g=g,
                         H=Hvec[ii], denseops=False)

    for jj in range(Nbodies):
        cylArray.add_body(x0[jj], y0[jj], device)

    cylArrays.append(cylArray)

    print('Finished array initialization')

    sys.stdout.flush()


###################### GRADIENT FLOW RUN #######################
# Initialize flowmap
flowmap = Flowmap(cylArrays, domain_constr, alpha_slam, min_dist,
                    adapt_tol=adapt_CG_tol, cg_tol=CG_tol)
w0 = flowmap.initial_state_and_global_scalings(x0, y0, gen_damping, gen_stiffness)

print('\nStarting time-stepping\n')
# Time stepping
if method=='RK12':
    result = eeheun(w0, flowmap, stop_tol,
                                      Tmax=Tfin, dt_0=dt_0,
                                      order=1, adapt_tol=adapt_RK_tol,
                                      atol=RK_atol, rtol=RK_rtol,
                                      monitor=monitor)
elif method=='EE':
    result = euler(w0, flowmap, stop_tol,
                                      Tmax=Tfin, dt_0=dt_0,
                                      monitor=monitor)
else:
    raise ValueError('Time stepping method can be either RK12 or EE')

if monitor:
    tvec, Wvec, fhist, ghist, monitordict, phinormvec = result
else:
    tvec, Wvec, fhist, ghist = result

# Save data
xhist = []
yhist = []
chist = []
khist = []
for ii in range(len(tvec)):
    x, y, _, c, k, _, _, _ = flowmap.w_split(Wvec[ii])
    xhist.append(x)
    yhist.append(y)
    chist.append(c)
    khist.append(k)


if monitor:
    np.savez(run_name, vertices=vertices, t=tvec, xhist=xhist, yhist=yhist,
             chist=chist, khist=khist, fhist=fhist, ghist=ghist,
             fscale=flowmap.cost_scale, phinormvec=phinormvec,
             monitor_tvec=monitordict['tvec'],
             monitor_timetotvec=monitordict['time_totvec'],
             monitor_timeCGvec=monitordict['time_CGvec'],
             monitor_CGitervec=monitordict['CGitervec'])
else:
    np.savez(run_name, vertices=vertices, t=tvec, xhist=xhist, yhist=yhist,
         chist=chist, khist=khist, fhist=fhist, ghist=ghist,
         fscale=flowmap.cost_scale)
