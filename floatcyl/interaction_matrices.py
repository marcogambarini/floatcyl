#!/usr/bin/env python3

import numpy as np
from scipy.special import *

def vector_index(n, m, Nn, Nm):
    """Computes the transformation from indices (m,n) to the single index
    of interaction vectors and matrices
    """
    return (n + Nn)*(Nm + 1) + m

def inverse_vector_indices(ii, Nn, Nm):
    """Computes the transformation from the single index
    of interaction vectors and matrices to indices (m,n)
    """
    n, m = divmod((ii - Nn*(Nm+1)),(Nm+1))

    return n, m


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



def body_diffraction_matrix(D, k, kq, d, a, Nn):
    """Computes the external diffraction matrix B from the results of the
    isolated body analysis stored in array D

    Parameters
    ----------
    D: 3-D array of single-body scattering results
    k: progressive wavenumber (real root of the dispersion equation)
    km: array of evanescent wavenumbers (imaginary roots of the
        dispersion equation), array of shape (Nm x 1)
    d: water depth
    a: radius
    Nn: number of angular/radial modes

    Returns
    -------
    B: array of shape ((2Nn + 1)(Nm + 1) x (2Nn + 1)(Nm + 1))
    """

    sqrtN0 = np.sqrt(0.5*(1 + (np.sinh(2*k*d)) / (2*k*d) ))
    sqrtNq = np.sqrt(0.5*(1 + (np.sin(2*kq*d)) / (2*kq*d) ))


    Nq = np.shape(kq)[0]

    B = np.zeros(((2*Nn + 1)*(Nq + 1),(2*Nn + 1)*(Nq + 1)), dtype=complex)


    for n in range(-Nn,Nn+1):
        #incident prog. wave, scattered prog. wave
        m = 0
        q = 0
        B[vector_index(n,q,Nn,Nq), vector_index(n,m,Nn,Nq)] = (
         hankel1(n,k*a)/jv(n,k*a)/h1vp(n,k*a) * (-jvp(n,k*a) +
         (1j)**(-n) * D[0,n,0]*np.cosh(k*d)/sqrtN0)
        )


        #incident prog. wave, scattered evan. wave
        m = 0
        for q in range(1,Nq+1):
            B[vector_index(n,q,Nn,Nq), vector_index(n,m,Nn,Nq)] = (
             kn(n,kq[q-1]*a)/jv(n,k*a) * (1j)**(-n) * D[q,n,0]/
             sqrtNq[q-1]/kvp(n, kq[q-1]*a)
            )

        #incident evan. wave, scattered prog. wave
        q = 0
        for m in range(1,Nq+1):
            B[vector_index(n,q,Nn,Nq), vector_index(n,m,Nn,Nq)] = (
             hankel1(n,k*a)/iv(n,kq[m-1]*a) * D[0,n,m]*np.cosh(k*d)/
             (sqrtN0 * h1vp(n,k*a))
            )

        #incident evan. wave, scattered evan. wave
        for q in range(1,Nq+1):
            for m in range(1,Nq+1):
                if (q!=m):
                    B[vector_index(n,q,Nn,Nq), vector_index(n,m,Nn,Nq)] = (
                     kn(n,kq[q-1]*a)/iv(n,kq[m-1]*a) *
                     D[q,n,m] / (sqrtNq[q-1]*kvp(n,kq[q-1]*a))
                    )
                else:
                    B[vector_index(n,q,Nn,Nq), vector_index(n,m,Nn,Nq)] = (
                     kn(n,kq[m-1]*a)/iv(n,kq[m-1]*a) * (
                     -ivp(n,kq[m-1]*a)/kvp(n,kq[m-1]*a) +
                     D[m,n,m]/(sqrtNq[m-1]*kvp(n,kq[m-1]*a))
                     )
                    )

    return B


def body_diffraction_matrix_internal(C, k, km, a, Nn):
    """Computes the internal diffraction matrix B from the results of the
    isolated body analysis stored in array C

    Parameters
    ----------
    C: 3-D array of single-body scattering results
    k: progressive wavenumber (real root of the dispersion equation)
    km: array of evanescent wavenumbers (imaginary roots of the
        dispersion equation), array of shape (Nm x 1)
    a: radius
    Nn: number of angular/radial modes


    Returns
    -------
    B: array of shape ((2Nn + 1)(Nm + 1) x (2Nn + 1)(Nm + 1))
    """

    Nm = np.shape(km)[0]
    B = np.zeros(((2*Nn + 1)*(Nm+1), (2*Nn + 1)*(Nm+1)), dtype=complex)


    for n in range(-Nn, Nn+1):
        m = 0
        s = 0
        B[vector_index(n, s, Nn, Nm), vector_index(n, m, Nn, Nm)] = (
         C[0, n, 0] / (2 * (1j)**n * jv(n, k*a))
        )

        m = 0
        for s in range(1, Nm+1):
            B[vector_index(n, s, Nn, Nm), vector_index(n, m, Nn, Nm)] = (
             C[s, n, 0] / ((1j)**n * jv(n, k*a))
            )

        s = 0
        for m in range(1, Nm+1):
            B[vector_index(n, s, Nn, Nm), vector_index(n, m, Nn, Nm)] = (
             C[0, n, m] / (2 * iv(n, km[m-1]*a))
            )

        for s in range(1, Nm+1):
            for m in range(1, Nm+1):
                B[vector_index(n, s, Nn, Nm), vector_index(n, m, Nn, Nm)] = (
                 C[s, n, m] / iv(n, km[m-1]*a)
                )


    return B


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


def heave_forces_basis(a, h, Nn, Nm):
    """Computes the integrals of the internal problem basis function on the
    lower surface of the cylinder. Heave forces are combinations of these terms.

    Parameters
    ----------
    a: cylinder radius
    h: cylinder clearance (elevation of lower surface over the sea bottom)
    Nn: number of angular/radial modes
    Nm: number of evanescent modes

    Returns
    -------
    Y: array of shape ((2Nn + 1)(Nm + 1) x 1)
    """

    Y = np.zeros(((2*Nn+1)*(Nm+1),1))

    m = 0
    n = 0
    Y[vector_index(n, m, Nn, Nm), 0] = np.pi * a * a

    for m in range(1, Nm+1):
        Y[vector_index(n, m, Nn, Nm), 0] = 1/i0(m*np.pi*a/h) * (
        ((-1)**m * 2 * a * h * i1(m*np.pi*a/h)) / m)

    return Y


def radiated_wave_coeffs(D, k, kq, d, a, Nl, Nq):
    """Computes the coefficients R of radiated wave coefficients
    for the external problem (useful for potential/free surface recovery)

    Parameters
    ----------
    D: vector of single-body radiation results
    k: progressive wavenumber (real root of the dispersion equation)
    kq: array of evanescent wavenumbers (imaginary roots of the
        dispersion equation), array of shape (Nm x 1)
    d: water depth
    a: radius
    Nl: number of angular/radial modes
    Nq: number of evanescent modes

    Returns
    -------
    R: array of shape ((2Nl + 1)(Nm + 1) x 1)
    """

    sqrtN0 = np.sqrt(0.5*(1 + (np.sinh(2*k*d)) / (2*k*d) ))
    sqrtNq = np.sqrt(0.5*(1 + (np.sin(2*kq*d)) / (2*kq*d) ))


    R = np.zeros(((2*Nl + 1)*(Nq+1), 1), dtype=complex)

    l = 0
    q = 0
    R[vector_index(l, q, Nl, Nq), 0] = hankel1(0, k*a) * (
                (D[0] * np.cosh(k*d)) / (h1vp(0, k*a) * sqrtN0))

    for q in range(1, Nq+1):
        R[vector_index(l, q, Nl, Nq), 0] = k0(kq[q-1]*a) * (
                D[q] / (kvp(0, kq[q-1]*a)*sqrtNq[q-1]))

    return R


def radiated_wave_coeffs_internal(C, k, kq, h, a, Nl, Ns):
    """Computes the coefficients Rtilde of radiated wave coefficients
    for the internal problem (useful for force computation)

    Parameters
    ----------
    C: vector of single-body radiation results
    k: progressive wavenumber (real root of the dispersion equation)
    kq: array of evanescent wavenumbers (imaginary roots of the
        dispersion equation), array of shape (Nm x 1)
    h: clearance (elevation of lower surface over the sea bottom)
    a: radius
    Nl: number of angular/radial modes
    Ns: number of evanescent modes

    Returns
    -------
    R: array of shape ((2Nl + 1)(Nm + 1) x 1)
    """

    R = np.zeros(((2*Nl + 1)*(Ns+1), 1), dtype=complex)

    l = 0
    s = 0
    R[vector_index(l, s, Nl, Ns), 0] = C[0]/2

    for s in range(1, Ns+1):
        R[vector_index(l, s, Nl, Ns), 0] = C[s]

    return R
