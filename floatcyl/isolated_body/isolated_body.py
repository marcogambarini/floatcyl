#!/usr/bin/env python3

import numpy as np
from scipy.special import *

from floatcyl.utils.utils import *

class Cylinder(object):

    def __init__(self, radius=1., draft=0.4, depth=30., k=0.01,
                kq=[0.4], Nn=5, Nq=10, omega=3.):
        self.radius = radius
        self.draft = draft
        self.depth = depth
        self.k = k
        self.kq = kq
        self.Nn = Nn
        self.Nq = Nq
        self.omega = omega

        self.clearance = depth-draft


    def compute_diffraction_properties(self):

        Nq = self.Nq
        Nn = self.Nn

        D = np.zeros((Nq+1, 2*Nn+1, Nq+1), dtype=complex) #external coeffs
        C = np.zeros((Nq+1, 2*Nn+1, Nq+1), dtype=complex) #internal coeffs

        for n in range(-Nn, Nn+1):
            for m in range(Nq+1):
                E, u, G = self.isolated_body_matrices_diff(n, m)

                #assemble and solve the system
                A = np.eye(Nq+1, dtype=complex) + E@G
                c_diff = np.linalg.solve(A, u)
                d_diff = G@c_diff

                D[:, n, m] = d_diff[:,0]
                C[:, n, m] = c_diff[:,0]

        #External diffraction matrix B
        B = self.body_diffraction_matrix(D)

        #Internal diffraction matrix Btilde
        Btilde = self.body_diffraction_matrix_internal(C)



    def isolated_body_matrices_diff(self, n, m):
        """Compute the isolated body diffraction matrices E, G and vector u

        Parameters
        ----------
        n: theta-mode
        m: z-mode (0 for incident progressive waves, >0 for incident evanescent waves)
        k: progressive wavenumber (real root of the dispersion equation)
        kq: array of evanescent wavenumbers (imaginary roots of the dispersion equation)
        h: clearance below the cylinder (depth-draft)
        d: depth
        a: cylinder radius

        Returns
        -------
        E: array of shape (Nq x Nq)
        u: array of shape (Nq x 1)
        G: array of shape (Nq x Nq)
        """

        k = self.k
        kq = self.kq
        h = self.clearance
        d = self.depth
        a = self.radius

        #initialization
        Nq = np.shape(kq)[0] + 1

        E = np.zeros((Nq, Nq), dtype=complex)
        u = np.zeros((Nq, 1), dtype=complex)
        G = np.zeros((Nq, Nq), dtype=complex)


        #computation of some repeatedly used intermediate results for speed
        sqrtN0 = np.sqrt(0.5*(1 + (np.sinh(2*k*d)) / (2*k*d) ))
        sqrtNq = np.sqrt(0.5*(1 + (np.sin(2*kq*d)) / (2*kq*d) ))
        sinhk0h = np.sinh(k*h)
        sinkqh = np.sin(kq*h)
        #Bessel functions computed with scipy.special
        dHnk0a = h1vp(n, k*a)
        dKnkqa = kvp(n, kq*a)



        #E matrix
        s_vector = np.arange(Nq)

        #zeroth column (progressive)
        E[:,0] = -2 * hankel1(n, k*a)/dHnk0a * ((h*k * sinhk0h) / sqrtN0) * (
            (-1)**s_vector / (s_vector*s_vector*np.pi*np.pi + k*k*h*h)
        )

        #create 2 grids of indices. sq[0,:,:] is s, sq[1,:,:] is q
        sq_grid = np.mgrid[0:Nq, 1:Nq]
        s = sq_grid[0,:,:]
        q = sq_grid[1,:,:]

        E[:,1:Nq] = -2*h*kv(n, kq[q-1][:,:,0]*a)/dKnkqa[q-1][:,:,0] * (
                    (kq[q-1][:,:,0]*(-1)**s*sinkqh[q-1][:,:,0]) /
                    (sqrtNq[q-1][:,:,0] * (-s*s*np.pi*np.pi + kq[q-1][:,:,0]*kq[q-1][:,:,0]*h*h)) )



        #U vector
        if (m==0):
            u[:,0] = 4*(1j)**(n+1)*(-1)**s_vector*h*sinhk0h / (np.pi*a*
                    (s_vector*s_vector*np.pi*np.pi + k*k*h*h) * dHnk0a * np.cosh(k*d))
        else:
            u[:,0] = 2*h*(-1)**(s_vector+1)*sinkqh[m-1] / (a * (-s_vector*s_vector*np.pi*np.pi +
                        kq[m-1]*kq[m-1]*h*h) * dKnkqa[m-1])


        #G matrix
        G[0, 0] = np.abs(n) * sinhk0h / (2*a*k*k*d*sqrtN0)

        q_vector = np.arange(1, Nq)
        G[1:Nq, 0] = np.abs(n) * sinkqh[q_vector-1][:,0] / (2*a*
                    kq[q_vector-1][:,0]*kq[q_vector-1][:,0]*d*sqrtNq[q_vector-1][:,0])

        s_vector = np.arange(1, Nq)
        G[0, 1:Nq] = ivp(n, s_vector*np.pi*a/h)/iv(n, s_vector*np.pi*a/h) * (
                    (s_vector*np.pi*h*(-1)**s_vector*sinhk0h)/((s_vector*s_vector*np.pi*np.pi + k*k*h*h)*d*sqrtN0))

        sq_grid = np.mgrid[1:Nq, 1:Nq]
        q = sq_grid[0,:,:]
        s = sq_grid[1,:,:]
        G[1:Nq, 1:Nq] = ivp(n, s*np.pi*a/h)/iv(n, s*np.pi*a/h) * (
                (s*np.pi*h*(-1)**s*sinkqh[q-1][:,:,0])/(
                (-s*s*np.pi*np.pi + kq[q-1][:,:,0]*kq[q-1][:,:,0]*h*h)*d*sqrtNq[q-1][:,:,0]))


        return E, u, G



    def isolated_body_matrices_rad(self, g=9.81):
        """Compute the isolated body radiation problem vectors q, s

        Parameters
        ----------
        k: progressive wavenumber (real root of the dispersion equation)
        kq: array of evanescent wavenumbers (imaginary roots of the dispersion equation)
        h: clearance below the cylinder (depth-draft)
        d: depth
        a: cylinder radius
        omega: wave frequency (rad/s)
        g: gravitational field

        Returns
        -------
        q: array of shape (Nq x 1)
        s: array of shape (Nq x 1)
        """

        k = self.k
        kq = self.kq
        h = self.clearance
        d = self.depth
        a = self.radius
        omega = self.omega


        #initialization
        Nq = np.shape(kq)[0] + 1

        q = np.zeros((Nq, 1), dtype=complex)
        s = np.zeros((Nq, 1), dtype=complex)

        sqrtN0 = np.sqrt(0.5*(1 + (np.sinh(2*k*d)) / (2*k*d) ))
        sqrtNq = np.sqrt(0.5*(1 + (np.sin(2*kq*d)) / (2*kq*d) ))


        q[0] = (1j)*omega*omega/(g*h) * (h*h/3 - 0.5*a*a)
        for ii in range(1, Nq):
            q[ii] = 2*(1j)*omega*omega*h*(-1)**ii/(g*ii*ii*np.pi*np.pi)


        s[0] = (1j)*a*omega*omega/(2*k*k*d*g*h*sqrtN0) * np.sinh(k*h)
        for ii in range(1, Nq):
            s[ii] = ((1j)*a*omega*omega/(2*kq[ii-1]*kq[ii-1]*d*g*h*sqrtNq[ii-1]) *
                            np.sin(kq[ii-1]*h))


        return q, s

    def body_diffraction_matrix(self, D):
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

        k = self.k
        kq = self.kq
        d = self.depth
        a = self.radius
        Nn = self.Nn

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


    def body_diffraction_matrix_internal(self, C):
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

        k = self.k
        km = self.kq
        a = self.radius
        Nn = self.Nn


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
