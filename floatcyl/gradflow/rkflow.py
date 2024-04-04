#! /usr/bin/env python3

import numpy as np
import sys

def eeheun(w0, flowmap, stop_tol, Tmax=1000, dt_0=1.9, order=1,
           atol=1e-6, rtol=1e-3, adapt_tol=True):
    """
    Explicit Euler/Heun adaptive time stepping

    Parameters
    ----------
    w0: array
        Initial guess
    flowmap: Flowmap object
        An object with methods compute_f_g and compute_phi
    stop_tol: float
        Stopping criterion (norm of phi)
    Tmax: float
        Maximum final time (default: 1000)
    dt_0: float
        Tentative time step (default: 1.9)
    order: integer
        Order of the method (= number of evaluations per time step)
        Either 1 (Euler update) or 2 (Heun update) (default: 1)
    atol: float
        Absolute tolerance for stepsize selection (default: 1e-6)
    rtol: float
        Relative tolerance for stepsize selection (default: 1e-3)
    adapt_tol: boolean
        Whether to adapt the stepsize selection tolerance to the
        stopping criterion (default: True). If True, values
        of atol and rtol will be ignored.

    Returns
    -------
    tvec: list of floats
        Vector of times
    Wvec: list of complex arrays
        Vector of control variable arrays along the time steps
    fhist: list of floats
        Vector of costs
    ghist: list of floats
        Vector containing the 2-norms of the constraint function
    """

    # Hardcoded safety factor values (see Hairer p. 167)
    # I don't think it's meaningful to let the user change them
    fac = 0.8
    facmin = 0.2
    facmax = 4

    stop_crit = 2*stop_tol
    t = 0
    dt = dt_0
    W = w0.copy()

    atol = None

    Wvec = [W]
    tvec = []
    fhist = []
    ghist = []

    f0, g_vec_0, k1 = flowmap.compute_phi(W)

    while t<Tmax and stop_crit>stop_tol:
        # output to log file now (don't wait)
        sys.stdout.flush()
        print('t = ', t)
        f, g_vec, k2 = flowmap.compute_phi(W + dt*k1)

        W_EE = W + dt * k1 # Explicit Euler step
        W_H = W + 0.5 * dt * (k1 + k2) # Heun step

        if adapt_tol:
            print('dt = ', dt)
            #atol = 0.1 * stop_tol * dt * np.linalg.norm(k1)/np.linalg.norm(k2 - k1)
            if atol is None:
                atol = 0.1 * dt * np.linalg.norm(k1)**2/np.linalg.norm(k2 - k1)
            else:
                atol = min(atol, 0.1 * dt * np.linalg.norm(k1)**2/np.linalg.norm(k2 - k1))
            print('Estimated required RK tolerance = ', atol)
            err = np.linalg.norm(W_H - W_EE) / atol
            print('Error indicator = ', err)
            print('k2-k1 = ', np.linalg.norm(k2 - k1))
        else:
            sc = atol + np.maximum(np.abs(W_EE), np.abs(W))*rtol
            err = 1/np.sqrt(len(W)) * np.linalg.norm(np.abs(W_H-W_EE)/sc)
        q = np.sqrt(1/err) # optimal stepsize factor
        dt = dt * min(facmax, max(facmin, fac*q)) # Hairer (4.13)
        if err <= 1: # stepsize accepted
            t += dt
            if order == 1: # for order 1 method with just an evaluation per step
                W = W_EE
                k1 = k2
            elif order == 2: # for order 2 method with 2 evaluations per step
                W = W_H
                f, g_vec, k1 = flowmap.compute_phi(W)

            normgvec = np.linalg.norm(g_vec, ord=2)
            Wvec.append(W)
            tvec.append(t)
            fhist.append(f)
            ghist.append(normgvec)
            stop_crit = np.linalg.norm(k1)

            print('\nTime ', t)
            print('Cost = ', f, ', constraint norm = ', ghist[-1])
            print('Stopping indicator = ', stop_crit)
        else:
            print('Stepsize rejected')

    return tvec, Wvec, fhist, ghist


def euler(w0, flowmap, stop_tol, Tmax=1000, dt_0=1.9):
    """
    Explicit Euler

    Parameters
    ----------
    w0: array
        Initial guess
    flowmap: Flowmap object
        An object with methods compute_f_g and compute_phi
    stop_tol: float
        Stopping criterion (norm of phi)
    Tmax: float
        Maximum final time (default: 1000)
    dt_0: float
        Tentative time step (default: 1.9)
    """

    stop_crit = 2*stop_tol
    t = 0
    dt = dt_0
    W = w0.copy()

    Wvec = [W]
    tvec = []
    fhist = []
    ghist = []


    while t<Tmax and stop_crit>stop_tol:
        # output to log file now (don't wait)
        sys.stdout.flush()
        dt = dt_0

        f0, g_vec0, phi = flowmap.compute_phi(W)
        W = W + dt * phi # Explicit Euler step

        normgvec = np.linalg.norm(g_vec0, ord=2)
        Wvec.append(W)
        tvec.append(t)
        fhist.append(f0)
        ghist.append(normgvec)
        stop_crit = np.linalg.norm(phi)

        print('\nTime ', t)
        print('Cost = ', f0, ', constraint norm = ', ghist[-1])
        print('Stopping indicator = ', stop_crit)

        t += dt

    return tvec, Wvec, fhist, ghist
