#!/usr/bin/env python3

import numpy as np
from scipy.special import *

from floatcyl.utils.utils import *

def basis_transformation_matrix(k, km, ai, aj, xi, yi, xj, yj, Nn, shutup=False):
    """Computes the transformation matrix T_ij from the incident wave
    basis of body i to the scattered wave basis of body j

    Parameters
    ----------
    k: progressive wavenumber (real root of the dispersion equation)
    km: array of evanescent wavenumbers (imaginary roots of the
        dispersion equation), array of shape (Nm x 1)
    ai: radius of cylinder i
    aj: radius of cylinder j
    xi, yi: coordinates of body i
    xj, yj: coordinates of body j
    Nn: number of angular/radial modes

    Returns
    -------
    Tij: array of shape ((2Nn + 1)(Nm + 1) x (2Nn + 1)(Nm + 1))
    """

    L_ij = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
    alpha_ij = np.arctan2((yj-yi),(xj-xi))

    if not shutup:
        print("L_ij = ", L_ij)
        print("alpha_ij = ", alpha_ij*180/np.pi, ' deg')

    Nm = np.shape(km)[0]

    T_ij = np.zeros(((2*Nn + 1)*(Nm + 1),(2*Nn + 1)*(Nm + 1)), dtype=complex)

     #print(np.shape(T_ij))

    for n in range(-Nn,Nn+1):
        for l in range(-Nn,Nn+1):
            #m=0
            m=0

            T_ij[vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                jv(l,k*aj)/hankel1(n,k*ai)*hankel1(n-l,k*L_ij)*
                np.exp(1j*alpha_ij*(n-l))
            )



            for m in range(1,Nm+1):

                T_ij[vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                iv(l,km[m-1]*aj)/kn(n,km[m-1]*ai)*kn(n-l,km[m-1]*L_ij)*
                np.exp(1j*alpha_ij*(n-l))*(-1)**l
                )

    return T_ij




def incident_wave_coeffs(k, beta, a, x, y, Nn, Nm):
    """Computes the incident wave coefficients for a body

    Parameters
    ----------
    k: progressive wavenumber (real root of the dispersion equation) (rad/s)
    beta: direction of incoming waves (rad)
    a: radius of the cylinder
    (x,y): coordinates of the center of the cylinder
    Nn: number of angular/radial modes
    Nm: number of evanescent modes

    Returns
    -------
    coeffs: array of shape ((2Nn + 1)(Nm + 1) x 1)
    """


    premult = np.exp(1j * k * (x*np.cos(beta) + y*np.sin(beta)))

    coeffs = np.zeros(((2*Nn+1)*(Nm+1), 1), dtype=complex)

    for n in range(-Nn, Nn+1):
        m = 0
        coeffs[vector_index(n, m, Nn, Nm), 0] = jv(n, k*a) * np.exp(1j * n * (np.pi/2 - beta))

    coeffs = coeffs*premult


    return coeffs


def scattered_basis_fs(n, m, k, depth, a, xc, yc, xeval, yeval):
    """Computes the scattered wave basis function of order (n,m) at the free surface

    Parameters
    ----------
    n: theta-mode
    m: z-mode (0 for incident progressive waves, >0 for incident evanescent waves)
    k: relevant wavenumber for the requested order (real root of the dispersion
    equation for m=0, m-th imaginary root of the dispersion equation for m>=1)
    depth: water depth
    a: cylinder radius
    xc, yc: center coordinates of the scattering body in the global reference
    xeval, yeval: (Npx1) arrays of coordinates of points where the basis function
    needs to be evaluated, in the global reference

    Returns
    psi: array of shape (Np x 1)

    """

    Np = np.shape(xeval)[0]
    psi = np.zeros(Np, dtype=complex)

    if (m==0):
        # for ii in range(Np):
        #     r = np.sqrt((xeval[ii]-xc)*(xeval[ii]-xc) + (yeval[ii]-yc)*(yeval[ii]-yc))
        #     theta = np.arctan2((yeval[ii]-yc),(xeval[ii]-xc))
        #
        #     psi[ii] = hankel1(n, k*r) * np.exp(1j*n*theta)

        r = np.sqrt((xeval-xc)*(xeval-xc) + (yeval-yc)*(yeval-yc))
        theta = np.arctan2((yeval-yc),(xeval-xc))
        psi = hankel1(n, k*r) * np.exp(1j*n*theta)

        psi = psi / hankel1(n, k*a)

    else:
        # for ii in range(Np):
        #     r = np.sqrt((xeval[ii]-xc)*(xeval[ii]-xc) + (yeval[ii]-yc)*(yeval[ii]-yc))
        #     theta = np.arctan2((yeval[ii]-yc),(xeval[ii]-xc))
        #
        #     psi[ii] = kn(n, k*r) * np.exp(1j*n*theta)

        r = np.sqrt((xeval-xc)*(xeval-xc) + (yeval-yc)*(yeval-yc))
        theta = np.arctan2((yeval-yc),(xeval-xc))
        psi = kn(n, k*r) * np.exp(1j*n*theta)

        psi = psi * np.cos(k*depth) / kn(n, k*a)


    return psi


def incident_basis_fs(n, m, k, depth, a, xc, yc, xeval, yeval):
    """Computes the incident wave basis function of order (n,m) at the free surface

    Parameters
    ----------
    n: theta-mode
    m: z-mode (0 for incident progressive waves, >0 for incident evanescent waves)
    k: relevant wavenumber for the requested order (real root of the dispersion
    equation for m=0, m-th imaginary root of the dispersion equation for m>=1)
    depth: water depth
    a: cylinder radius
    xc, yc: center coordinates of the scattering body in the global reference
    xeval, yeval: (Npx1) arrays of coordinates of points where the basis function
    needs to be evaluated, in the global reference

    Returns
    psi: array of shape (Np x 1)

    """

    Np = np.shape(xeval)[0]
    psi = np.zeros(Np, dtype=complex)

    if (m==0):
        for ii in range(Np):
            r = np.sqrt((xeval[ii]-xc)*(xeval[ii]-xc) + (yeval[ii]-yc)*(yeval[ii]-yc))
            theta = np.arctan2((yeval[ii]-yc),(xeval[ii]-xc))

            psi[ii] = jv(n, k*r) * np.exp(1j*n*theta)

        psi = psi / jv(n, k*a)

    else:
        for ii in range(Np):
            r = np.sqrt((xeval[ii]-xc)*(xeval[ii]-xc) + (yeval[ii]-yc)*(yeval[ii]-yc))
            theta = np.arctan2((yeval[ii]-yc),(xeval[ii]-xc))

            psi[ii] = iv(n, k*r) * np.exp(1j*n*theta)

        psi = psi * np.cos(k*depth) / iv(n, k*a)


    return psi
