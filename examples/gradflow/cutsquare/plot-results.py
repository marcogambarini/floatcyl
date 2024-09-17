#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from projector import *
import sys

try:
    plt.style.use('latexsimple')
except:
    pass

runFolder = sys.argv[1]
fileName = sys.argv[2]

data = np.load(runFolder+'/'+fileName+'.npz', allow_pickle=True)
t = data['t']
x = data['xhist']
y = data['yhist']
k = data['khist']
c = data['chist']
f = data['fhist']
g = data['ghist']
vertices = data['vertices']
phinormvec = data['phinormvec']

monitor_tvec = data['monitor_tvec']
monitor_timetotvec = data['monitor_timetotvec']
monitor_timeCGvec = data['monitor_timeCGvec']
monitor_CGitervec = data['monitor_CGitervec']

L = 50
draft = 0.5
depth = 30.

proj = PolyProjector(vertices)


niter = len(t)
nbodies = x.shape[1]

# compute the time step
dt = np.diff(t)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, f)
ax[1].semilogy(t, g)
ax[0].set_ylabel(r'$f$')
ax[1].set_ylabel(r'$\|g\|$')
ax[1].set_xlabel('Fictitious time')
plt.savefig(runFolder+'/'+fileName+'-fg.pdf')

# Plot just the positions
fig, ax = plt.subplots()
proj.plot(ax)
for ii in range(nbodies):
    circle = plt.Circle((x[0,ii], y[0,ii]), radius=2, color='r')
    ax.add_patch(circle)
#ax.set_title('Initial configuration')
ax.set_aspect('equal')
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$y$ (m)')
plt.savefig(runFolder+'/'+fileName+'-initconf.pdf')

fig, ax = plt.subplots()
proj.plot(ax)
for ii in range(nbodies):
    circle = plt.Circle((x[-1,ii], y[-1,ii]), radius=2, color='r')
    ax.add_patch(circle)
#ax.set_title('Optimization result')
ax.set_aspect('equal')
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$y$ (m)')
plt.savefig(runFolder+'/'+fileName+'-optconf.pdf')

# Plot positions and controls
import matplotlib as mpl
cmap = mpl.colormaps['Reds']

def scatter_labels(x, y, v, cmap, ax, s=0, vmin=None, vmax=None):
    aa = ax.scatter(x, y, s=s, c=v, cmap=cmap, vmin=vmin, vmax=vmax)

    for ii in range(len(x)):
        ax.text(x[ii], y[ii], format(v[ii], '.0f'), fontsize=12,
                bbox={'boxstyle' : 'circle', 'edgecolor' : aa.to_rgba(v[ii]),
                      'facecolor' : 'white', 'linewidth' : 3.5},
                ha='center', va='center')

    return aa

fig, ax = plt.subplots()
#fig, ax = plt.subplots()
aa = scatter_labels(x[-1,:], y[-1,:], c[-1,:], cmap, ax)
cbar = plt.colorbar(aa)
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$y$ (m)')
ax.set_aspect('equal')
cbar.set_label('Damping (N/(m/s))')
plt.savefig(runFolder+'/'+fileName+'-damp.pdf')

fig, ax = plt.subplots()
#fig, ax = plt.subplots()
aa = scatter_labels(x[-1,:], y[-1,:], k[-1,:], cmap, ax)
cbar = plt.colorbar(aa)
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$y$ (m)')
ax.set_aspect('equal')
cbar.set_label('Stiffness (N/m)')
plt.savefig(runFolder+'/'+fileName+'-stiff.pdf')

fig, ax = plt.subplots(2, 1)
ax[0].plot(monitor_timetotvec, label='Total')
ax[0].plot(monitor_timeCGvec, label='CG')
ax[1].plot(monitor_CGitervec)
ax[1].set_xlabel('Calls')
ax[0].set_ylabel('Time per call (s)')
ax[1].set_ylabel('Iterations')
ax[0].legend(loc='center right')
plt.savefig(runFolder+'/'+fileName+'-monitor.pdf')

fig, ax = plt.subplots(2, 1)
ax[0].plot(t[:-1], dt)
ax[1].semilogy(t, phinormvec)
ax[1].set_xlabel('Fictitious time')
ax[0].set_ylabel('Time step')
ax[1].set_ylabel(r'$\|\bm{\Psi}\|$')
plt.savefig(runFolder+'/'+fileName+'-dtphi.pdf')

fig, ax = plt.subplots(2, 1)
ax[0].semilogy(t[:-1], np.abs(np.diff(x, axis=0)))
ax[0].semilogy(t[:-1], np.abs(np.diff(y, axis=0)))
ax[1].semilogy(t[:-1], np.abs(np.diff(k, axis=0)))
ax[1].semilogy(t[:-1], np.abs(np.diff(c, axis=0)))
plt.savefig(runFolder+'/'+fileName+'-controlvar.pdf')

#plt.show()
