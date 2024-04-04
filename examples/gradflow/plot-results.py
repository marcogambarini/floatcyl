#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from projector import *

try:
    plt.style.use('latexsimple')
except:
    pass

fileName = 'test1.npz'

data = np.load(fileName, allow_pickle=True)
t = data['t']
x = data['xhist']
y = data['yhist']
k = data['khist']
c = data['chist']
f = data['fhist']
g = data['ghist']
vertices = data['vertices']

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

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(t[:-1], dt)
ax[1].plot(t, f)
#ax[2].plot(t, g)
ax[2].semilogy(t, g)
ax[2].set_xlabel('fictitious time')
ax[0].set_ylabel('dt')
ax[1].set_ylabel('f')
ax[2].set_ylabel('g')

# Plot just the positions
fig, ax = plt.subplots()
proj.plot(ax)
for ii in range(nbodies):
    circle = plt.Circle((x[0,ii], y[0,ii]), radius=2, color='r')
    ax.add_patch(circle)
ax.set_title('Initial configuration')
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

fig, ax = plt.subplots()
proj.plot(ax)
for ii in range(nbodies):
    circle = plt.Circle((x[-1,ii], y[-1,ii]), radius=2, color='r')
    ax.add_patch(circle)
ax.set_title('Optimization result')
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

# Plot positions and controls
import matplotlib
cmap = matplotlib.cm.get_cmap('Reds')

def scatter_labels(x, y, v, cmap, ax, s=0, vmin=None, vmax=None):
    aa = ax.scatter(x, y, s=s, c=v, cmap=cmap, vmin=vmin, vmax=vmax)

    for ii in range(len(x)):
        ax.text(x[ii], y[ii], format(v[ii], '.0f'), fontsize=12,
                bbox={'boxstyle' : 'circle', 'edgecolor' : aa.to_rgba(v[ii]),
                      'facecolor' : 'white', 'linewidth' : 3.5},
                ha='center', va='center')

    return aa

fig, ax = plt.subplots(figsize=(4.5,6))
#fig, ax = plt.subplots()
aa = scatter_labels(x[-1,:], y[-1,:], c[-1,:], cmap, ax)
cbar = plt.colorbar(aa)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_aspect('equal')
# ax.set_xlim(-5,25)
# ax.set_ylim(-5,25)
cbar.set_label('Damping (N/(m/s))')

fig, ax = plt.subplots(figsize=(4.5,6))
#fig, ax = plt.subplots()
aa = scatter_labels(x[-1,:], y[-1,:], k[-1,:], cmap, ax)
cbar = plt.colorbar(aa)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_aspect('equal')
# ax.set_xlim(-5,25)
# ax.set_ylim(-5,25)
cbar.set_label('Stiffness (N/m)')

fig, ax = plt.subplots(2, 1)
ax[0].plot(monitor_tvec, monitor_timetotvec)
ax[0].plot(monitor_tvec, monitor_timeCGvec)
ax[1].plot(monitor_tvec, monitor_CGitervec)

plt.show()
