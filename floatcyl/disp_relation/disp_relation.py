#!/usr/bin/env python3

import numpy as np
from scipy.optimize import fsolve, bisect


def real_disp_rel(omega, depth, g=9.81):
    """Compute the real wavenumber for a finite depth gravity wave of frequency omega

    Parameters
    ----------
    omega: angular frequency of the wave (rad/s)
    depth: water depth (m)
    g: gravitational field (m/s^2), default=9.81

    Returns
    -------
    wavenumber (rad/m)
    """

    def disp_rel(x):
        return x * np.tanh(x*depth) - omega*omega/g

    #initial guess: shallow-water
    x0 = omega/np.sqrt(g*depth)
    k = fsolve(disp_rel, x0)

    return k


def imag_disp_rel(omega, depth, N, g=9.81):
    """Compute the first N complex wavenumbers for a finite depth gravity wave of frequency omega

    Parameters
    ----------
    omega: angular frequency of the wave (rad/s)
    depth: water depth (m)
    N: number of wavenumbers to compute
    g: gravitational field (m/s^2), default=9.81

    Returns
    -------
    array of shape (Nx1)
        wavenumber (rad/m)
    """

    def disp_rel(x):
        return x * np.tan(x*depth) + omega*omega/g

    k = np.zeros((N,1))
    for n in range(N):
        #because of the periodicity of the tangent function
        xL = ( np.pi/2*(1+1e-6) + n * np.pi) / depth
        xR = ( np.pi/2*(1-1e-6) + (n+1) * np.pi) / depth
        k[n] = bisect(disp_rel, xL, xR)

    return k
