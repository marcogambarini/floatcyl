#!/usr/bin/env python3

import numpy as np
from scipy.special import *
from scipy.sparse import coo_matrix

from floatcyl.utils.utils import *

class Array(object):

    def __init__(self, beta=0., depth=30., k=0.01,
                kq=[0.4], Nn=5, Nq=10, omega=3., water_density=1000.,
                g = 9.81, denseops = False):
        """Constructor

        Parameters
        ----------
        beta: float
            incoming wave direction
        depth: float
            sea depth (default: 30.)
        k: float
            real wavenumber (default: 0.01)
        kq: list or array of floats
            imaginary wavenumbers (default: [0.4])
        Nn: int
            number of progressive (real) modes (default: 5)
        Nq: int
            number of evanescent (imaginary) modes (default: 10)
        omega: float
            radial frequency of the waves (default: 3)
        water_density: float
            water density (default: 1000)
        g: float
            gravitational field (default: 9.81)
        denseops: boolean
            whether to use dense matrices and direct solvers
            (default: False, will use sparse matrices and GMRES)
        """
        self.beta = beta
        self.depth = depth
        self.k = k
        self.kq = kq
        self.Nn = Nn
        self.Nq = Nq
        self.omega = omega
        self.water_density = water_density
        self.g = g

        self.bodies = []
        self.x = []
        self.y = []
        self.Nbodies = 0


    def add_body(self, xbody, ybody, body):
        """Adds single body to the array.
        When a body is added, it inherits all the properties
        that are common to the whole array.

        Parameters
        ----------
        xbody: float
            x coordinate of body center
        ybody: float
            y coordinate of body center
        body: body object
        """
        body.depth = self.depth
        body.k = self.k
        body.kq = self.kq
        body.Nn = self.Nn
        body.Nq = self.Nq
        body.omega = self.omega
        body.water_density = self.water_density
        body.g = self.g
        body.clearance = self.depth-body.draft

        self.x.append(xbody)
        self.y.append(ybody)
        self.bodies.append(body)

        self.Nbodies = self.Nbodies+1

    def solve(self):
        """Assembles and solves the full array system.
        See Eq. (4.11) - (4.15) in Child 2011.
        """


        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)

        Nbodies = self.Nbodies

        M11 = -np.eye(Nnq*Nbodies, dtype=complex)
        M12 = np.zeros((Nnq*Nbodies, Nbodies), dtype=complex)
        M21 = np.zeros((Nbodies, Nnq*Nbodies), dtype=complex)
        M22 = np.eye(Nbodies, dtype=complex)

        h1 = np.zeros((Nnq*Nbodies, 1), dtype=complex)
        h2 = np.zeros((Nbodies, 1), dtype=complex)

        for ii in range(Nbodies):
            k = self.k
            beta = self.beta
            a = self.bodies[ii].radius
            inc_wave_coeffs = self.incident_wave_coeffs(k, beta, a, self.x[ii], self.y[ii], Nn, Nq)
            B = self.bodies[ii].B
            W = self.bodies[ii].W
            Btilde = self.bodies[ii].Btilde
            Y = self.bodies[ii].Y
            h1[ii*Nnq:(ii+1)*Nnq,:] = -B@inc_wave_coeffs
            h2[ii] = -1/W * inc_wave_coeffs.T @ Btilde.T @ Y
            #h2[ii] = (-1j)*-1/W * inc_wave_coeffs.T @ Btilde.T @ Y

            for jj in range(Nbodies):
                if not (ii==jj):
                    Tij = self.basis_transformation_matrix(jj, ii, shutup=True)

                    B = self.bodies[jj].B
                    R = self.bodies[ii].R
                    Btilde = self.bodies[jj].Btilde
                    Y = self.bodies[jj].Y

                    #block matrix filling of matrix M11
                    M11[ii*Nnq:(ii+1)*Nnq, jj*Nnq:(jj+1)*Nnq] = (B @ Tij.T).toarray()
                    self.M11 = M11 #for debugging and checking the spectral radius

                    #block column-vector filling of matrix M12
                    M12[ii*Nnq:(ii+1)*Nnq, jj] = (B @ Tij.T @ R)[:,0]

                    #block row-vector filling of matrix M21
                    M21[ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (Tij @ Btilde.T @ Y).T)[0,:]

                    #elementwise filling of matrix M22
                    M22[ii, jj] = 1/W * R.T @ Tij @ Btilde.T @ Y

        # Save blocks for use in optimization routines
        self.M11 = M11
        self.M12 = M12
        self.M21 = M21
        self.M22 = M22

        self.h1 = h1
        self.h2 = h2


        # Build matrix M and vector h from blocks
        M = np.block([[M11, M12],[M21,M22]])
        hh = np.block([[h1],[h2]])



        # Solve the system
        z = np.linalg.solve(M, hh)
        self.scatter_coeffs = z[:Nnq*Nbodies]
        self.rao = z[Nnq*Nbodies:]


    def compute_free_surface(self, x, y):
        """Computes the free surface complex elevation for the fully
        coupled problem (incident+diffracted+radiated waves)

        Parameters
        ----------
        x, y: floats
            grid coordinates in x, y

        Returns
        -------
        eta: complex array of shape (shape(y), shape(x))
        """

        X, Y = np.meshgrid(x, y)
        nx = np.shape(x)[0]
        ny = np.shape(y)[0]
        Xflat = np.matrix.flatten(X)
        Yflat = np.matrix.flatten(Y)

        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)

        k = self.k
        kq = self.kq
        Nbodies = self.Nbodies
        xc = self.x
        yc = self.y

        #contribution of incident waves
        eta_incident = np.exp(1j*k*(Xflat*np.cos(self.beta) +
                                Yflat*np.sin(self.beta)))

        #scattering contribution
        eta_s = np.zeros(np.shape(Xflat), dtype=complex)

        for ii in range(Nbodies):
            a = self.bodies[ii].radius
            etai = np.zeros(np.shape(Xflat), dtype=complex)
            for ll in range(Nnq):
                n, m = inverse_vector_indices(ll, Nn, Nq)
                if m==0:
                    etai = etai + (self.scatter_coeffs[Nnq*ii + ll] *
                            self.scattered_basis_fs(n,m,k,a,xc[ii],yc[ii],Xflat,Yflat))
                else:
                    etai = etai + (self.scatter_coeffs[Nnq*ii + ll] *
                            self.scattered_basis_fs(n,m,kq[m-1],a,xc[ii],yc[ii],Xflat,Yflat))

            eta_s = eta_s + etai


        #radiation contribution
        eta_r = np.zeros(np.shape(Xflat), dtype=complex)

        for ii in range(Nbodies):
            a = self.bodies[ii].radius
            R = self.bodies[ii].R
            eta_r_i = np.zeros(np.shape(Xflat), dtype=complex)

            for ll in range(np.shape(R)[0]):
                n, m = inverse_vector_indices(ll, Nn, Nq)
                if n==0:
                    if m==0:
                        eta_r_i = eta_r_i + (R[ll] *
                                self.scattered_basis_fs(n,m,k,a,xc[ii],yc[ii],Xflat,Yflat))
                    else:
                        eta_r_i = eta_r_i + (R[ll] *
                                self.scattered_basis_fs(n,m,kq[m-1],a,xc[ii],yc[ii],Xflat,Yflat))


            eta_r_i = eta_r_i * self.rao[ii]
            eta_r = eta_r + eta_r_i

        #eta_r[:,Xflat**2 + Yflat**2 < a**2] = np.nan

        eta_incident = eta_incident.reshape(ny, nx)
        eta_s = eta_s.reshape(ny, nx)
        eta_r = eta_r.reshape(ny, nx)



        return eta_s+eta_incident+eta_r


    def basis_transformation_matrix(self, ii, jj, shutup=False):
        """Computes the transformation matrix T_ij from the incident wave
        basis of body i to the scattered wave basis of body j.
        See Eq. (3.119) in Child 2011.
        Parameters
        ----------
        ii: integer
            Index of receiving body
        jj: integer
            Index of emitting body
        shutup: boolean
            Whether or not to print an output to screen
        Returns
        -------
        Tij: array of shape ((2Nn + 1)(Nm + 1) x (2Nn + 1)(Nm + 1))
        """

        k = self.k
        km = self.kq
        Nn = self.Nn
        ai = self.bodies[ii].radius
        aj = self.bodies[jj].radius
        xi = self.x[ii]
        yi = self.y[ii]
        xj = self.x[jj]
        yj = self.y[jj]

        L_ij = np.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj))
        alpha_ij = np.arctan2((yj-yi),(xj-xi))

        if not shutup:
            print("i = ", ii, ", j = ", jj)
            print("L_ij = ", L_ij)
            print("alpha_ij = ", alpha_ij*180/np.pi, ' deg')

        Nm = np.shape(km)[0]

        #T_ij = np.zeros(((2*Nn + 1)*(Nm + 1),(2*Nn + 1)*(Nm + 1)), dtype=complex)

        # precompute vectors
        jv_vec = jv(np.arange(-Nn, Nn+1), k*aj)
        hank_vec = hankel1(np.arange(-Nn, Nn+1), k*ai)
        hank_diff_vec = (hankel1(np.arange(-2*Nn, 2*Nn+1), k*L_ij) *
                        np.exp(1j*alpha_ij*(np.arange(-2*Nn, 2*Nn+1))))
        M, L = np.meshgrid(np.arange(Nm), np.arange(-Nn, Nn+1))
        iv_mat = iv(L,km[M][:,:,0]*aj)
        kn_mat = kn(L,km[M][:,:,0]*ai)
        M, LL = np.meshgrid(np.arange(Nm), np.arange(-2*Nn, 2*Nn+1))
        kn_diff_mat = kn(LL,km[M][:,:,0]*L_ij)
        exp_vec = np.exp(1j*alpha_ij*np.arange(-2*Nn, 2*Nn+1))

        # vectors for building sparse matrix in coordinate form
        data = np.zeros((2*Nn+1)**2*(Nm+1), dtype=complex)
        row_ind = np.zeros((2*Nn+1)**2*(Nm+1), dtype=np.int32)
        col_ind = np.zeros((2*Nn+1)**2*(Nm+1), dtype=np.int32)

        counter = 0

        for n in range(-Nn,Nn+1):
            for l in range(-Nn,Nn+1):
                #m=0
                m=0

                data[counter] = jv_vec[l+Nn]/hank_vec[n+Nn]*hank_diff_vec[n-l+2*Nn]
                row_ind[counter] = vector_index(n,m,Nn,Nm)
                col_ind[counter] = vector_index(l,m,Nn,Nm)
                counter += 1

                data[counter:counter+Nm] = (iv_mat[l+Nn,:]/kn_mat[n+Nn,:]*kn_diff_mat[n-l+2*Nn,:]*
                            exp_vec[n-l+2*Nn]*(-1)**l)


                for m in range(1,Nm+1):
                    #data[counter] = (iv_mat[l+Nn,m-1]/kn_mat[n+Nn,m-1]*kn_diff_mat[n-l+2*Nn,m-1]*
                    #            exp_vec[n-l+2*Nn]*(-1)**l)
                    row_ind[counter] = vector_index(n,m,Nn,Nm)
                    col_ind[counter] = vector_index(l,m,Nn,Nm)
                    counter += 1

        T_ij = coo_matrix((data, (row_ind, col_ind)))


        return T_ij



    def incident_wave_coeffs(self, k, beta, a, x, y, Nn, Nm):
        """Computes the incident wave coefficients for a body.
        See Eq. (3.114) in Child 2011.

        Parameters
        ----------
        k: float
            progressive wavenumber (real root of the dispersion equation) (rad/s)
        beta: float
            direction of incoming waves (rad)
        a: float
            radius of the cylinder
        x, y: floats
            coordinates of the center of the cylinder
        Nn: int
            number of angular/radial modes
        Nm: int
            number of evanescent modes

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


    def scattered_basis_fs(self, n, m, k, a, xc, yc, xeval, yeval):
        """Computes the scattered wave basis function of order (n,m) at the free surface.
        See Eq. (3.110) in Child 2011.


        Parameters
        ----------
        n: int
            theta-mode
        m: int
            z-mode (0 for incident progressive waves, >0 for incident evanescent waves)
        k: float
            relevant wavenumber for the requested order (real root of the dispersion
            equation for m=0, m-th imaginary root of the dispersion equation for m>=1)
            depth: water depth
        a: float
            cylinder radius
        xc, yc: floats
            center coordinates of the scattering body in the global reference
        xeval, yeval: ints
            (Npx1) arrays of coordinates of points where the basis function
            needs to be evaluated, in the global reference

        Returns
        -------
        psi: array of shape (Np x 1)

        """
        depth = self.depth

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


    def incident_basis_fs(self, n, m, k, depth, a, xc, yc, xeval, yeval):
        """Computes the incident wave basis function of order (n,m) at the free surface.
        See Eq. (3.111) in Child 2011.

        Parameters
        ----------
        n: int
            theta-mode
        m: int
            z-mode (0 for incident progressive waves, >0 for incident evanescent waves)
        k: float
            relevant wavenumber for the requested order (real root of the dispersion
            equation for m=0, m-th imaginary root of the dispersion equation for m>=1)
            depth: water depth
        a: float
            cylinder radius
        xc, yc: floats
            center coordinates of the scattering body in the global reference
        xeval, yeval: (Npx1) arrays of floats
            coordinates of points where the basis function
            needs to be evaluated, in the global reference

        Returns
        -------
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




    def L_derivatives(self):
        """
        Computes the derivatives of the distance and angle between pairs
        of bodies with respect to their coordinates.
        See Eq. (4.23)-(4.27) in Gallizioli 2022.
        """

        x_coord = self.x
        y_coord = self.y
        Nb = self.Nbodies

        L = np.zeros((Nb,Nb))
        alpha = np.zeros((Nb,Nb))

        dL_dxi = np.zeros((Nb,Nb))
        dL_dxj = np.zeros((Nb,Nb))
        dL_dyi = np.zeros((Nb,Nb))
        dL_dyj = np.zeros((Nb,Nb))
        dalpha_dxi = np.zeros((Nb,Nb))
        dalpha_dxj = np.zeros((Nb,Nb))
        dalpha_dyi = np.zeros((Nb,Nb))
        dalpha_dyj = np.zeros((Nb,Nb))

        for ii in range(Nb):
            for jj in range(Nb):
                if ii!=jj:

                    L[ii,jj] = np.sqrt((x_coord[jj] - x_coord[ii])**2 + (y_coord[jj] - y_coord[ii])**2)
                    alpha[ii,jj] = np.arctan2(y_coord[jj] - y_coord[ii], x_coord[jj] - x_coord[ii])

                    M = np.array(((np.cos(alpha[ii,jj]), -L[ii,jj]*np.sin(alpha[ii,jj])),
                                  (np.sin(alpha[ii,jj]),  L[ii,jj]*np.cos(alpha[ii,jj]))))

                    b = [-1, 0]
                    x = np.linalg.solve(M, b)
                    dL_dxi[ii,jj] = x[0]
                    dalpha_dxi[ii,jj] = x[1]

                    b = [1, 0]
                    x = np.linalg.solve(M, b)
                    dL_dxj[ii,jj] = x[0]
                    dalpha_dxj[ii,jj] = x[1]

                    b = [0, -1]
                    x = np.linalg.solve(M, b)
                    dL_dyi[ii,jj] = x[0]
                    dalpha_dyi[ii,jj] = x[1]

                    b = [0, 1]
                    x = np.linalg.solve(M, b)
                    dL_dyj[ii,jj] = x[0]
                    dalpha_dyj[ii,jj] = x[1]


        return L, alpha, dL_dxi, dL_dxj, dL_dyi, dL_dyj, dalpha_dxi, dalpha_dxj, dalpha_dyi, dalpha_dyj


    def T_derivatives(self):
        """
        Computes the derivatives of the basis transformation matrices
        with respect to the coordinates of bodies.
        See Eq. (4.20) in Gallizioli 2022
        """
        Nn = self.Nn
        Nm = self.Nq
        a = self.bodies[0].radius

        k = self.k
        km = self.kq

        Nb = self.Nbodies

        Lder = self.L_derivatives()

        L = Lder[0]
        alpha = Lder[1]

        dT_dxi = np.zeros((Nb,Nb,(2*Nn + 1)*(Nm + 1),(2*Nn + 1)*(Nm + 1)), dtype=complex)
        dT_dxj = np.zeros((Nb,Nb,(2*Nn + 1)*(Nm + 1),(2*Nn + 1)*(Nm + 1)), dtype=complex)

        dT_dyi = np.zeros((Nb,Nb,(2*Nn + 1)*(Nm + 1),(2*Nn + 1)*(Nm + 1)), dtype=complex)
        dT_dyj = np.zeros((Nb,Nb,(2*Nn + 1)*(Nm + 1),(2*Nn + 1)*(Nm + 1)), dtype=complex)


        for ii in range(Nb):
            for jj in range(Nb):
                if ii!=jj :

                    for n in range(-Nn,Nn+1):
                        for l in range(-Nn,Nn+1):
                            m=0
                            jv_h1 = jv(l,k*a)/hankel1(n,k*a)
                            dT_dL = ((n-l)*hankel1(n-l,k*L[ii,jj])/(k*L[ii,jj])-hankel1(n-l+1,k*L[ii,jj]))*np.exp(1j*alpha[ii,jj]*(n-l))*k
                            dT_da = hankel1(n-l,k*L[ii,jj])*np.exp(1j*alpha[ii,jj]*(n-l))*1j*(n-l)

                            dT_dxi[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                                        jv_h1 * (dT_dL * Lder[2][ii,jj]
                                        + dT_da * Lder[6][ii,jj])
                                        )

                            dT_dyi[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                                        jv_h1 * (dT_dL * Lder[4][ii,jj]
                                        + dT_da * Lder[8][ii,jj])
                                        )

                            dT_dxj[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                                        jv_h1 * (dT_dL * Lder[3][ii,jj]
                                        + dT_da * Lder[7][ii,jj])
                                        )

                            dT_dyj[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                                        jv_h1 * (dT_dL * Lder[5][ii,jj]
                                        + dT_da * Lder[9][ii,jj])
                                        )



                            for m in range(1,Nm+1):

                                iv_kn = iv(l,km[m-1]*a)/kn(n,km[m-1]*a)
                                dT_dL = (-0.5*(kn(n-l-1,km[m-1]*L[ii,jj])+kn(n-l+1,km[m-1]*L[ii,jj]))
                                            *np.exp(1j*alpha[ii,jj]*(n-l))*(-1)**l*km[m-1])
                                dT_da = kn(n-l,km[m-1]*L[ii,jj])*np.exp(1j*alpha[ii,jj]*(n-l))*(-1)**l*1j*(n-l)


                                dT_dxi[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                                iv_kn * (dT_dL * Lder[2][ii,jj]
                                + dT_da * Lder[6][ii,jj])
                                )

                                dT_dyi[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                                iv_kn * (dT_dL * Lder[4][ii,jj]
                                + dT_da * Lder[8][ii,jj])
                                )

                                dT_dxj[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                                iv_kn * (dT_dL * Lder[3][ii,jj]
                                + dT_da * Lder[7][ii,jj])
                                )

                                dT_dyj[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                                iv_kn * (dT_dL * Lder[5][ii,jj]
                                + dT_da * Lder[9][ii,jj])
                                )


        return dT_dxi, dT_dxj, dT_dyi, dT_dyj


    def adjoint_equations(self):
        """
        Solves the adjoint equations.
        See (4.14) of Gallizioli 2022.
        """
        Nbodies = self.Nbodies
        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)

        M11 = self.M11
        M12 = self.M12
        M21 = self.M21
        M22 = self.M22

        rao = self.rao

        C = np.zeros((Nbodies,Nbodies))
        for ii in range(Nbodies):
            C[ii,ii] = self.bodies[ii].gamma

        h1 = np.zeros((Nnq*Nbodies, 1), dtype=complex)
        h2 = self.omega**2 * C@rao

        M11H = M11.conj().T
        M12H = M12.conj().T
        M21H = M21.conj().T
        M22H = M22.conj().T

        MH = np.block([[M11H, M21H],[M12H,M22H]])
        mulan = np.block([[h1],[h2]])

        # Solve the system
        z = np.linalg.solve(MH, mulan)
        self.landa = z[:Nnq*Nbodies]
        self.mu = z[Nnq*Nbodies:]


    def M_derivatives(self):
        """
        Computes the derivatives of the matrix blocks of the primal
        system (needed to assemble the gradient) with respect to the
        positions of the bodies.
        See Eq. (4.16)-(4.20) of Gallizioli 2022
        """
        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)

        Nbodies = self.Nbodies

        dM11_dxi = np.zeros((Nbodies,Nnq*Nbodies,Nnq*Nbodies), dtype=complex)
        dM12_dxi = np.zeros((Nbodies,Nnq*Nbodies, Nbodies), dtype=complex)
        dM21_dxi = np.zeros((Nbodies,Nbodies, Nnq*Nbodies), dtype=complex)
        dM22_dxi = np.zeros((Nbodies,Nbodies,Nbodies), dtype=complex)

        dM11_dyi = np.zeros((Nbodies,Nnq*Nbodies,Nnq*Nbodies), dtype=complex)
        dM12_dyi = np.zeros((Nbodies,Nnq*Nbodies, Nbodies), dtype=complex)
        dM21_dyi = np.zeros((Nbodies,Nbodies, Nnq*Nbodies), dtype=complex)
        dM22_dyi = np.zeros((Nbodies,Nbodies,Nbodies), dtype=complex)

        dT = self.T_derivatives()
        dT_dxi = dT[0]
        dT_dxj = dT[1]
        dT_dyi = dT[2]
        dT_dyj = dT[3]

        for kk in range(Nbodies):

            for ii in range(Nbodies):
                for jj in range(Nbodies):
                    if ii!=jj:
                        B = self.bodies[ii].B
                        R = self.bodies[jj].R
                        Btilde = self.bodies[ii].Btilde
                        Y = self.bodies[ii].Y
                        W = self.bodies[ii].W

                        if kk==ii:
                            ###
                            dM11_dxi[kk][ii*Nnq:(ii+1)*Nnq, jj*Nnq:(jj+1)*Nnq] = B @ dT_dxj[jj,kk].T

                            dM12_dxi[kk][ii*Nnq:(ii+1)*Nnq, jj] = (B @ dT_dxj[jj,kk].T @ R)[:,0]

                            dM21_dxi[kk][ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (dT_dxj[jj,kk] @ Btilde.T @ Y).T)[0,:]

                            dM22_dxi[kk][ii, jj] = 1/W * R.T @ dT_dxj[jj,kk] @ Btilde.T @ Y


                            ###
                            dM11_dyi[kk][ii*Nnq:(ii+1)*Nnq, jj*Nnq:(jj+1)*Nnq] = B @ dT_dyj[jj,kk].T

                            dM12_dyi[kk][ii*Nnq:(ii+1)*Nnq, jj] = (B @ dT_dyj[jj,kk].T @ R)[:,0]

                            dM21_dyi[kk][ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (dT_dyj[jj,kk] @ Btilde.T @ Y).T)[0,:]

                            dM22_dyi[kk][ii, jj] = 1/W * R.T @ dT_dyj[jj,kk] @ Btilde.T @ Y


                        if kk==jj:
                            ###
                            dM11_dxi[kk][ii*Nnq:(ii+1)*Nnq, jj*Nnq:(jj+1)*Nnq] = B @ dT_dxi[kk,ii].T

                            dM12_dxi[kk][ii*Nnq:(ii+1)*Nnq, jj] = (B @ dT_dxi[kk,ii].T @ R)[:,0]

                            dM21_dxi[kk][ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (dT_dxi[kk,ii] @ Btilde.T @ Y).T)[0,:]

                            dM22_dxi[kk][ii, jj] = 1/W * R.T @ dT_dxi[kk,ii] @ Btilde.T @ Y

                            ###
                            dM11_dyi[kk][ii*Nnq:(ii+1)*Nnq, jj*Nnq:(jj+1)*Nnq] = B @ dT_dyi[kk,ii].T

                            dM12_dyi[kk][ii*Nnq:(ii+1)*Nnq, jj] = (B @ dT_dyi[kk,ii].T @ R)[:,0]

                            dM21_dyi[kk][ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (dT_dyi[kk,ii] @ Btilde.T @ Y).T)[0,:]

                            dM22_dyi[kk][ii, jj] = 1/W * R.T @ dT_dyi[kk,ii] @ Btilde.T @ Y




        return dM11_dxi, dM12_dxi, dM21_dxi, dM22_dxi, dM11_dyi, dM12_dyi, dM21_dyi, dM22_dyi

    def h_derivatives(self):
        """
        Computes the derivatives of the rhs blocks of the primal system
        (needed to assemble the gradient) with respect to the
        positions of the bodies.
        Amended on 20/04/23 (TODO: document!)
        """
        k = self.k[0]
        beta = self.beta
        Nbodies = self.Nbodies
        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)

        dh1_dxi = np.zeros((Nbodies, Nnq*Nbodies, 1), dtype=complex)
        dh2_dxi = np.zeros((Nbodies, Nbodies, 1), dtype=complex)
        dh1_dyi = np.zeros((Nbodies, Nnq*Nbodies, 1), dtype=complex)
        dh2_dyi = np.zeros((Nbodies, Nbodies, 1), dtype=complex)

        for ii in range(Nbodies):
            a = self.bodies[ii].radius
            inc_wave_coeffs = self.incident_wave_coeffs(k, beta, a, self.x[ii], self.y[ii], Nn, Nq)
            B = self.bodies[ii].B
            W = self.bodies[ii].W
            Btilde = self.bodies[ii].Btilde
            Y = self.bodies[ii].Y

            dh1_dxi[ii,ii*Nnq:(ii+1)*Nnq,:] = -1j*k*np.cos(beta)*B@inc_wave_coeffs
            dh2_dxi[ii,ii] = -1j*k*np.cos(beta)*1/W * inc_wave_coeffs.T @ Btilde.T @ Y
            dh1_dyi[ii,ii*Nnq:(ii+1)*Nnq,:] = -1j*k*np.sin(beta)*B@inc_wave_coeffs
            dh2_dyi[ii,ii] = -1j*k*np.sin(beta)*1/W * inc_wave_coeffs.T @ Btilde.T @ Y

        return dh1_dxi, dh2_dxi, dh1_dyi, dh2_dyi


    def gradientJ(self):
        """
        Computes the derivatives of the Lagrangian with respect to
        the positions of the bodies.
        See Eq. (4.15) of Gallizioli 2022
        """
        dM = self.M_derivatives()
        Nbodies = self.Nbodies
        A = self.scatter_coeffs
        rao = self.rao
        k = self.k
        h1 = self.h1
        h2 = self.h2
        beta = self.beta

        dM11_dxi = dM[0]
        dM12_dxi = dM[1]
        dM21_dxi = dM[2]
        dM22_dxi = dM[3]

        dM11_dyi = dM[4]
        dM12_dyi = dM[5]
        dM21_dyi = dM[6]
        dM22_dyi = dM[7]

        dh = self.h_derivatives()

        dh1_dxi = dh[0]
        dh2_dxi = dh[1]
        dh1_dyi = dh[2]
        dh2_dyi = dh[3]

        landa = self.landa
        mu = self.mu

        dL_dxi = np.zeros(Nbodies)
        dL_dyi = np.zeros(Nbodies)

        for ii in range(Nbodies):

            dL_dxi[ii] = np.real(landa.conj().T@(dM11_dxi[ii]@A + dM12_dxi[ii]@rao -
                                                    dh1_dxi[ii]) +
                                 mu.conj().T@(dM21_dxi[ii]@A + dM22_dxi[ii]@rao -
                                                    dh2_dxi[ii]) )
            dL_dyi[ii] = np.real(landa.conj().T@(dM11_dyi[ii]@A + dM12_dyi[ii]@rao -
                                                    dh1_dyi[ii]) +
                                 mu.conj().T@(dM21_dyi[ii]@A + dM22_dyi[ii]@rao -
                                                    dh2_dyi[ii]) )


        gradJx = dL_dxi
        gradJy = dL_dyi

        return gradJx, gradJy


    def M_derivatives_damp(self):
        """
        Computes the derivatives of the system matrices with respect to
        the damping coefficient.
        See Eq. (4.36)-(4.37) in Gallizioli 2022
        """
        g = self.g
        rho = self.water_density
        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)
        omega = self.omega
        Nbodies = self.Nbodies

        dM21_dci = np.zeros((Nbodies,Nbodies, Nnq*Nbodies), dtype=complex)
        dM22_dci = np.zeros((Nbodies,Nbodies,Nbodies), dtype=complex)

        for ii in range(Nbodies): #loop on bodies
            W = self.bodies[ii].W
            Btilde = self.bodies[ii].Btilde
            Y = self.bodies[ii].Y

            for jj in range(Nbodies): # loop on columns
                for kk in range(Nbodies): # loop on rows
                    if jj!=kk:
                        Tij = self.basis_transformation_matrix(jj, kk, shutup=True)

                        R = self.bodies[jj].R
                        Btilde = self.bodies[kk].Btilde
                        Y = self.bodies[kk].Y
                        W = self.bodies[kk].W
                        if ii==kk:

                            dM21_dci[ii][kk, jj*Nnq:(jj+1)*Nnq] = -1/(W**2)*omega/(rho*g)*((Tij@Btilde.T@Y).T)
                            dM22_dci[ii][kk, jj] = -1/(W**2)*(R.T@Tij@Btilde.T@Y)*omega/(rho*g)

        return dM21_dci, dM22_dci


    def gradientJ_damp(self):
        """
        Computes the gradient of the Lagrangian with respect to the
        damping coefficients.
        See Eq. (4.35), (4.38) of Gallizioli 2022.
        """
        rao = self.rao
        A = self.scatter_coeffs

        omega = self.omega
        Nbodies = self.Nbodies
        mu = self.mu
        Nn = self.Nn
        Nq = self.Nq
        k = self.k
        beta = self.beta
        g = self.g
        rho = self.water_density
        Mder = self.M_derivatives_damp()
        dM21_dci = Mder[0]
        dM22_dci = Mder[1]

        DP = np.zeros(Nbodies)
        Dh = np.zeros((Nbodies,Nbodies), dtype=complex)
        dL_dci = np.zeros(Nbodies)


        for ii in range(Nbodies):
            a = self.bodies[ii].radius
            inc_wave_coeffs = self.incident_wave_coeffs(k, beta, a, self.x[ii], self.y[ii], Nn, Nq)
            W = self.bodies[ii].W
            Y = self.bodies[ii].Y
            DP[ii] = np.real(-0.5 *omega**2 *rao[ii].conj().T@rao[ii])

            for jj in range(Nbodies):
                if jj==ii:
                    Btilde = self.bodies[ii].Btilde
                    Dh[ii,jj] = 1/(W**2) * inc_wave_coeffs.T @ Btilde.T @ Y *omega / (rho*g)

        for ii in range(Nbodies):
            dL_dci[ii] = (DP[ii]
                +np.real(mu.conj().T@(dM21_dci[ii]@A
                +dM22_dci[ii]@rao)-mu.conj().T@Dh[ii]))

        return dL_dci
