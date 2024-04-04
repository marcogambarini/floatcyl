#! /usr/bin/env python3

import numpy as np
import configparser
import csv
import floatcyl as fcl
from floatcyl.gradflow.random_pos import random_in_square
from floatcyl.gradflow.constr_jac import PolyConstraint
from floatcyl.gradflow.flowmap import Flowmap
from floatcyl.gradflow.rkflow import *

#################### READ PARAMETER FILE ####################
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
conf.read('params.ini')

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

Te = conf['waves'].getfloat('Te')
Hs = conf['waves'].getfloat('Hs')
beta = conf['waves'].getfloat('beta') # in degrees
thres = conf['waves'].getfloat('thres')
Nf = conf['waves'].getint('Nf')
crit = conf['waves'].get('crit')

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

######################## SPECTRUM SETUP ########################
omegavec, amplitude = fcl.utils.utils.discrete_PM_spectrum(
                            Te, Hs, Nf, filename=None, thres=thres,
                            plot=False, crit=crit)
Hvec = 2*amplitude

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


###################### GRADIENT FLOW RUN #######################
# Initialize flowmap
flowmap = Flowmap(cylArrays, domain_constr, alpha_slam, min_dist,
                    adapt_tol=adapt_CG_tol, cg_tol=CG_tol)
w0 = flowmap.initial_state_and_global_scalings(x0, y0, gen_damping, gen_stiffness)

# Time stepping
if method=='RK12':
    tvec, Wvec, fhist, ghist = eeheun(w0, flowmap, stop_tol,
                                      Tmax=Tfin, dt_0=dt_0,
                                      order=1, adapt_tol=adapt_RK_tol,
                                      atol=RK_atol, rtol=RK_rtol)
elif method=='EE':
    tvec, Wvec, fhist, ghist = euler(w0, flowmap, stop_tol,
                                      Tmax=Tfin, dt_0=dt_0)
else:
    raise ValueError('Time stepping method can be either RK12 or EE')


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

np.savez(run_name, vertices=vertices, t=tvec, xhist=xhist, yhist=yhist,
         chist=chist, khist=khist, fhist=fhist, ghist=ghist,
         fscale=flowmap.cost_scale)
