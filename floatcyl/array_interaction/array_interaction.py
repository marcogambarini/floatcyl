#!/usr/bin/env python3

import numpy as np
from scipy.special import *
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, gmres

from floatcyl.utils.utils import *


class Array(object):

    def __init__(self, beta=0., depth=30., k=0.01,
                kq=[0.4], Nn=5, Nq=10, omega=3., water_density=1000.,
                g = 9.81, H=2., denseops = False):
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
        H: float
            monochromatic wave height
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
        self.H = H
        self.damping = None

        self.bodies = []
        self.x = []
        self.y = []
        self.W = []
        self.Nbodies = 0

        self.denseops = denseops


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
        try:
            self.W.append(body.W)
        except:
            print('Diffraction and radiation properties of bodies missing.'+
            '\nDoing the computation now...')
            body.compute_diffraction_properties()
            body.compute_radiation_properties()
            print('... done')
            self.W.append(body.W)

        self.Nbodies = self.Nbodies+1

    def update_controls(self, damping, stiffness):
        """
        Updates the values of vector self.W with specified damping and
        stiffness vectors

        Parameters
        ----------
        damping: float array of size Nbodies
            Damping coefficients
        stiffness: float array of size Nbodies
            Stiffness coefficients
        """

        if (len(damping) is not self.Nbodies
            or len(stiffness) is not self.Nbodies):
            raise ValueError('Input vectors should have a length equal to the number of bodies!')


        self.damping = damping
        rho = self.water_density
        omega = self.omega

        for ii in range(self.Nbodies):
            self.W[ii] = (self.bodies[ii].equiv_area
                        - 1j/(rho*self.g) * (self.bodies[ii].mass*omega*omega
                                - self.bodies[ii].hyd_stiff
                                + 1j*omega*damping[ii] - stiffness[ii]))

    def expose_operators(self):
        """Exposes the system operator, its hermitian and the rhs of
        the system. Sparse matrices + linear operators are used.
        """

        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)

        Nbodies = self.Nbodies

        M12 = np.zeros((Nnq*Nbodies, Nbodies), dtype=complex)
        M21 = np.zeros((Nbodies, Nnq*Nbodies), dtype=complex)
        M22 = np.eye(Nbodies, dtype=complex)

        h1 = np.zeros((Nnq*Nbodies, 1), dtype=complex)
        h2 = np.zeros((Nbodies, 1), dtype=complex)

        Tij = {}

        for ii in range(Nbodies):
            k = self.k
            beta = self.beta
            a = self.bodies[ii].radius
            inc_wave_coeffs = self.incident_wave_coeffs(k, beta, a, self.x[ii], self.y[ii], Nn, Nq)
            B = self.bodies[ii].B
            W = self.W[ii]
            Btilde = self.bodies[ii].Btilde
            Y = self.bodies[ii].Y
            h1[ii*Nnq:(ii+1)*Nnq,:] = -B@inc_wave_coeffs
            h2[ii] = -1/W * inc_wave_coeffs.T @ Btilde.T @ Y
            #h2[ii] = (-1j)*-1/W * inc_wave_coeffs.T @ Btilde.T @ Y

            for jj in range(Nbodies):
                if not (ii==jj):
                    Tij[ii, jj] = self.basis_transformation_matrix(jj, ii, shutup=True)

                    R = self.bodies[jj].R

                    #block column-vector filling of matrix M12
                    M12[ii*Nnq:(ii+1)*Nnq, jj] = (B @ (Tij[ii, jj].T @ R))[:,0]

                    #block row-vector filling of matrix M21
                    M21[ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (Tij[ii, jj] @ (Btilde.T @ Y)).T)[0,:]

                    #elementwise filling of matrix M22
                    M22[ii, jj] = 1/W * (R.T @ Tij[ii, jj]) @ (Btilde.T @ Y)

        # Save blocks for use in optimization routines

        def M11v(v):
            mv = np.zeros(len(v), dtype=complex)
            for ii in range(Nbodies):
                for jj in range(Nbodies):
                    if not (ii==jj):
                        mv[ii*Nnq:(ii+1)*Nnq] += (
                            self.bodies[ii].B @ (Tij[ii, jj].T @ v[jj*Nnq:(jj+1)*Nnq]))

            # identity contribution in return
            return -v + mv

        M11 = LinearOperator((Nnq*Nbodies, Nnq*Nbodies), matvec=M11v)


        # Build matrix M and vector h from blocks
        def Mv(v):
            mv = np.zeros(len(v), dtype=complex)
            mv[:Nnq*Nbodies] += M11@v[:Nnq*Nbodies]
            mv[:Nnq*Nbodies] += M12@v[Nnq*Nbodies:]
            mv[Nnq*Nbodies:] += M21@v[:Nnq*Nbodies]
            mv[Nnq*Nbodies:] += M22@v[Nnq*Nbodies:]

            return mv

        M = LinearOperator(((Nnq+1)*Nbodies, (Nnq+1)*Nbodies), matvec=Mv)
        hh = np.block([[h1],[h2]])

        def M11Hv(v):
            mv = np.zeros(len(v), dtype=complex)
            for ii in range(Nbodies):
                for jj in range(Nbodies):
                    if not (ii==jj):
                        mv[jj*Nnq:(jj+1)*Nnq] += (
                            np.conj(Tij[ii, jj]) @ (np.conj(self.bodies[ii].B.T) @ v[ii*Nnq:(ii+1)*Nnq]))

            # identity contribution in return
            return -v + mv

        M11H = LinearOperator((Nnq*Nbodies, Nnq*Nbodies), matvec=M11Hv)

        def MHv(v):
            mv = np.zeros(len(v), dtype=complex)
            mv[:Nnq*Nbodies] += M11H@v[:Nnq*Nbodies]
            mv[:Nnq*Nbodies] += np.conj(M21.T)@v[Nnq*Nbodies:]
            mv[Nnq*Nbodies:] += np.conj(M12.T)@v[:Nnq*Nbodies]
            mv[Nnq*Nbodies:] += np.conj(M22.T)@v[Nnq*Nbodies:]

            return mv

        MH = LinearOperator(((Nnq+1)*Nbodies, (Nnq+1)*Nbodies), matvec=MHv)

        return M, MH, hh[:,0]


    def solve(self):
        """Assembles and solves the full array system.
        See Eq. (4.11) - (4.15) in Child 2011.
        """


        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)

        Nbodies = self.Nbodies

        if self.denseops:

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
                W = self.W[ii]
                Btilde = self.bodies[ii].Btilde
                Y = self.bodies[ii].Y
                h1[ii*Nnq:(ii+1)*Nnq,:] = -B@inc_wave_coeffs
                h2[ii] = -1/W * inc_wave_coeffs.T @ Btilde.T @ Y
                #h2[ii] = (-1j)*-1/W * inc_wave_coeffs.T @ Btilde.T @ Y

                for jj in range(Nbodies):
                    if not (ii==jj):
                        Tij = self.basis_transformation_matrix(jj, ii, shutup=True)

                        R = self.bodies[jj].R

                        #block matrix filling of matrix M11
                        M11[ii*Nnq:(ii+1)*Nnq, jj*Nnq:(jj+1)*Nnq] = (B @ Tij.T).toarray()

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

        else:
            # sparse operations
            M12 = np.zeros((Nnq*Nbodies, Nbodies), dtype=complex)
            M21 = np.zeros((Nbodies, Nnq*Nbodies), dtype=complex)
            M22 = np.eye(Nbodies, dtype=complex)

            h1 = np.zeros((Nnq*Nbodies, 1), dtype=complex)
            h2 = np.zeros((Nbodies, 1), dtype=complex)

            Tij = {}

            for ii in range(Nbodies):
                k = self.k
                beta = self.beta
                a = self.bodies[ii].radius
                inc_wave_coeffs = self.incident_wave_coeffs(k, beta, a, self.x[ii], self.y[ii], Nn, Nq)
                B = self.bodies[ii].B
                W = self.W[ii]
                Btilde = self.bodies[ii].Btilde
                Y = self.bodies[ii].Y
                h1[ii*Nnq:(ii+1)*Nnq,:] = -B@inc_wave_coeffs
                h2[ii] = -1/W * inc_wave_coeffs.T @ Btilde.T @ Y
                #h2[ii] = (-1j)*-1/W * inc_wave_coeffs.T @ Btilde.T @ Y

                for jj in range(Nbodies):
                    if not (ii==jj):
                        Tij[ii, jj] = self.basis_transformation_matrix(jj, ii, shutup=True)

                        R = self.bodies[jj].R

                        #block column-vector filling of matrix M12
                        M12[ii*Nnq:(ii+1)*Nnq, jj] = (B @ (Tij[ii, jj].T @ R))[:,0]

                        #block row-vector filling of matrix M21
                        M21[ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (Tij[ii, jj] @ (Btilde.T @ Y)).T)[0,:]

                        #elementwise filling of matrix M22
                        M22[ii, jj] = 1/W * (R.T @ Tij[ii, jj]) @ (Btilde.T @ Y)

            # Save blocks for use in optimization routines

            def M11v(v):
                mv = np.zeros(len(v), dtype=complex)
                for ii in range(Nbodies):
                    for jj in range(Nbodies):
                        if not (ii==jj):
                            mv[ii*Nnq:(ii+1)*Nnq] += (
                                self.bodies[ii].B @ (Tij[ii, jj].T @ v[jj*Nnq:(jj+1)*Nnq]))

                # identity contribution in return
                return -v + mv

            M11 = LinearOperator((Nnq*Nbodies, Nnq*Nbodies), matvec=M11v)

            self.M11 = M11
            self.M12 = M12
            self.M21 = M21
            self.M22 = M22
            self.Tij = Tij

            self.h1 = h1
            self.h2 = h2


            # Build matrix M and vector h from blocks
            def Mv(v):
                mv = np.zeros(len(v), dtype=complex)
                mv[:Nnq*Nbodies] += M11@v[:Nnq*Nbodies]
                mv[:Nnq*Nbodies] += M12@v[Nnq*Nbodies:]
                mv[Nnq*Nbodies:] += M21@v[:Nnq*Nbodies]
                mv[Nnq*Nbodies:] += M22@v[Nnq*Nbodies:]

                return mv

            M = LinearOperator(((Nnq+1)*Nbodies, (Nnq+1)*Nbodies), matvec=Mv)
            hh = np.block([[h1],[h2]])



            # Solve the system
            z, info = gmres(M, hh)
            self.scatter_coeffs = np.zeros((Nnq*Nbodies, 1), dtype=complex)
            self.scatter_coeffs[:,0] = z[:Nnq*Nbodies]
            self.rao = np.zeros((Nbodies, 1), dtype=complex)
            self.rao[:,0] = z[Nnq*Nbodies:]


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

        if L_ij < (ai + aj):
            raise RuntimeError('Overlapping bodies: cannot compute the basis transformation matrix')

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


        return self.H/2*coeffs


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


    def compute_power(self, individual=False, OWC=False):
        """
        Computes the power of the array.

        Parameters
        ----------
        individual: boolean
            Whether to return a list of the powers of the single devices
            (true) or just the total power (false)

        Return
        ------
        P: float (if individual=False) or array (if individual=True)
            Power
        """
        Nbodies = self.Nbodies
        rao = self.rao
        omega = self.omega
        bodies = self.bodies

        P_individual = np.zeros(Nbodies)

        for ii in range(Nbodies):
            b = bodies[ii].torque_coeff
            if OWC:
                for jj in range(len(b)):
                    P_individual[ii] += b[jj] * omega**(2*jj) * (np.abs(rao[ii]))**(2*jj)
            else:
                if self.damping is not None:
                    P_individual[ii] = (
                        0.5 * self.damping[ii] * omega**2 * np.abs(rao[ii])**2 )
                else:
                    P_individual[ii] = (
                        0.5 * bodies[ii].gamma * omega**2 * np.abs(rao[ii])**2 )

        if individual:
            return P_individual
        else:
            return np.sum(P_individual)


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

        k = self.k
        km = self.kq

        Nb = self.Nbodies

        Lder = self.L_derivatives()

        dist = Lder[0]
        alpha = Lder[1]

        # dT_dxi = np.zeros((Nb,Nb,(2*Nn + 1)*(Nm + 1),(2*Nn + 1)*(Nm + 1)), dtype=complex)
        # dT_dxj = np.zeros((Nb,Nb,(2*Nn + 1)*(Nm + 1),(2*Nn + 1)*(Nm + 1)), dtype=complex)
        #
        # dT_dyi = np.zeros((Nb,Nb,(2*Nn + 1)*(Nm + 1),(2*Nn + 1)*(Nm + 1)), dtype=complex)
        # dT_dyj = np.zeros((Nb,Nb,(2*Nn + 1)*(Nm + 1),(2*Nn + 1)*(Nm + 1)), dtype=complex)

        dT_dxi = {}
        dT_dxj = {}

        dT_dyi = {}
        dT_dyj = {}

        for ii in range(Nb):
            for jj in range(Nb):
                if ii!=jj :

                    ai = self.bodies[ii].radius
                    aj = self.bodies[jj].radius


                    jv_vec = jv(np.arange(-Nn, Nn+1), k*aj)
                    hank_vec = hankel1(np.arange(-Nn, Nn+1), k*ai)
                    hank_diff_vec = hankel1(np.arange(-2*Nn, 2*Nn+2), k*dist[ii,jj])
                    M, L = np.meshgrid(np.arange(Nm), np.arange(-Nn, Nn+1))
                    iv_mat = iv(L,km[M][:,:,0]*aj)
                    kn_mat = kn(L,km[M][:,:,0]*ai)
                    M, LL = np.meshgrid(np.arange(Nm), np.arange(-2*Nn, 2*Nn+2))
                    kn_diff_mat = kn(LL,km[M][:,:,0]*dist[ii,jj])
                    exp_vec = np.exp(1j*alpha[ii,jj]*np.arange(-2*Nn, 2*Nn+1))


                    dT_dxi_data = np.zeros((2*Nn+1)**2*(Nm+1), dtype=complex)

                    dT_dxj_data = np.zeros((2*Nn+1)**2*(Nm+1), dtype=complex)

                    dT_dyi_data = np.zeros((2*Nn+1)**2*(Nm+1), dtype=complex)

                    dT_dyj_data = np.zeros((2*Nn+1)**2*(Nm+1), dtype=complex)

                    row_ind = np.zeros((2*Nn+1)**2*(Nm+1), dtype=np.int32)
                    col_ind = np.zeros((2*Nn+1)**2*(Nm+1), dtype=np.int32)

                    counter = 0

                    for n in range(-Nn,Nn+1):
                        for l in range(-Nn,Nn+1):
                            m=0
                            jv_h1 = jv_vec[l+Nn]/hank_vec[n+Nn]
                            dT_dL = ((n-l)*hank_diff_vec[n-l+2*Nn]/(k*dist[ii,jj])-hank_diff_vec[n-l+1+2*Nn])*exp_vec[n-l+2*Nn]*k
                            dT_da = hank_diff_vec[n-l+2*Nn]*exp_vec[n-l+2*Nn]*1j*(n-l)

                            # dT_dxi[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                            #             jv_h1 * (dT_dL * Lder[2][ii,jj]
                            #             + dT_da * Lder[6][ii,jj])
                            #             )
                            dT_dxi_data[counter] = (
                                        jv_h1 * (dT_dL * Lder[2][ii,jj]
                                        + dT_da * Lder[6][ii,jj])
                                        )
                            row_ind[counter] = vector_index(n,m,Nn,Nm)
                            col_ind[counter] = vector_index(l,m,Nn,Nm)

                            # dT_dyi[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                            #             jv_h1 * (dT_dL * Lder[4][ii,jj]
                            #             + dT_da * Lder[8][ii,jj])
                            #             )
                            dT_dyi_data[counter] = (
                                        jv_h1 * (dT_dL * Lder[4][ii,jj]
                                        + dT_da * Lder[8][ii,jj])
                                        )

                            # dT_dxj[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                            #             jv_h1 * (dT_dL * Lder[3][ii,jj]
                            #             + dT_da * Lder[7][ii,jj])
                            #             )
                            dT_dxj_data[counter] = (
                                        jv_h1 * (dT_dL * Lder[3][ii,jj]
                                        + dT_da * Lder[7][ii,jj])
                                        )

                            # dT_dyj[ii,jj][vector_index(n,m,Nn,Nm), vector_index(l,m,Nn,Nm)] = (
                            #             jv_h1 * (dT_dL * Lder[5][ii,jj]
                            #             + dT_da * Lder[9][ii,jj])
                            #             )
                            dT_dyj_data[counter] = (
                                        jv_h1 * (dT_dL * Lder[5][ii,jj]
                                        + dT_da * Lder[9][ii,jj])
                                        )

                            # precomputations so that loops are vectorized by scipy
                            #iv_kn_vec = iv(l,km*a)/kn(n,km*a)
                            iv_kn_vec = (iv_mat[l+Nn, :]/kn_mat[n+Nn, :]).reshape((Nm, 1))
                            #dT_dL_vec = (-0.5*(kn(n-l-1,km*dist[ii,jj])+kn(n-l+1,km*dist[ii,jj]))
                            #            *np.exp(1j*alpha[ii,jj]*(n-l))*(-1)**l*km)
                            dT_dL_vec = (-0.5*(kn_diff_mat[n-l-1+2*Nn,:]+kn_diff_mat[n-l+1+2*Nn,:]).reshape((Nm, 1))
                                        *exp_vec[n-l+2*Nn]*(-1)**l*km)
                            dT_da_vec = kn_diff_mat[n-l+2*Nn,:].reshape((Nm, 1))*exp_vec[n-l+2*Nn]*(-1)**l*1j*(n-l)

                            dT_dxi_vec = (iv_kn_vec * (dT_dL_vec * Lder[2][ii,jj]
                                + dT_da_vec * Lder[6][ii,jj]))
                            dT_dyi_vec = (iv_kn_vec * (dT_dL_vec * Lder[4][ii,jj]
                                + dT_da_vec * Lder[8][ii,jj]))
                            dT_dxj_vec = (iv_kn_vec * (dT_dL_vec * Lder[3][ii,jj]
                                + dT_da_vec * Lder[7][ii,jj]))
                            dT_dyj_vec = (iv_kn_vec * (dT_dL_vec * Lder[5][ii,jj]
                                + dT_da_vec * Lder[9][ii,jj]))

                            counter +=1

                            indn = vector_index(n, np.arange(1,Nm+1), Nn, Nm)
                            indl = vector_index(l, np.arange(1,Nm+1), Nn, Nm)

                            row_ind[counter:counter+Nm] = indn
                            col_ind[counter:counter+Nm] = indl

                            dT_dxi_data[counter:counter+Nm] = dT_dxi_vec[:,0]
                            dT_dyi_data[counter:counter+Nm] = dT_dyi_vec[:,0]
                            dT_dxj_data[counter:counter+Nm] = dT_dxj_vec[:,0]
                            dT_dyj_data[counter:counter+Nm] = dT_dyj_vec[:,0]

                            counter += Nm


                    dT_dxi[ii, jj] = coo_matrix((dT_dxi_data, (row_ind, col_ind)))
                    dT_dxj[ii, jj] = coo_matrix((dT_dxj_data, (row_ind, col_ind)))
                    dT_dyi[ii, jj] = coo_matrix((dT_dyi_data, (row_ind, col_ind)))
                    dT_dyj[ii, jj] = coo_matrix((dT_dyj_data, (row_ind, col_ind)))

        return dT_dxi, dT_dxj, dT_dyi, dT_dyj


    def adjoint_equations(self, OWC=False, mu=0, zmax=None):
        """
        Solves the adjoint equations.
        See (4.14) of Gallizioli 2022.

        Parameters
        ----------
        OWC: boolean
            Whether to use the formulation of the RHS for oscillating water
            columns with immersed turbine (default: False)
        mu: float
            Penalty parameter for constraint enforcement (default: 0,
            no additional constraints)
        zmax: float
            Maximum stroke amplitude for enforcement through the penalty
            method (only effective if mu>0)
        """
        Nbodies = self.Nbodies
        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)

        rao = self.rao

        print('rao = ', rao)

        bodies = self.bodies

        if self.damping is None:
            C = np.zeros((Nbodies,Nbodies))
            for ii in range(Nbodies):
                C[ii,ii] = bodies[ii].gamma
        else:
            C = np.diag(self.damping)

        h1 = np.zeros((Nnq*Nbodies, 1), dtype=complex)
        if OWC:
            h2 = np.zeros((Nbodies, 1), dtype=complex)
            for ii in range(Nbodies):
                b = bodies[ii].torque_coeff
                z = rao[ii]
                for jj in range(len(b)):
                    h2[ii] += (b[jj] * 2*jj * self.omega**(2*jj) *
                                np.conj(z)**(jj-1) * z**jj)
        else:
            h2 = self.omega**2 * C@rao

        # constraint enforcement via penalty method
        if mu>0:
            h2_penalty = -2*mu*rao * np.maximum(0, np.abs(rao)**2 - zmax**2)
            h2 += h2_penalty
            print('penalty term = ', np.linalg.norm(h2_penalty))

        print('h2 = ', h2)

        mulan = np.block([[h1],[h2]])

        if self.denseops:
            M11 = self.M11
            M12 = self.M12
            M21 = self.M21
            M22 = self.M22



            M11H = M11.conj().T
            M12H = M12.conj().T
            M21H = M21.conj().T
            M22H = M22.conj().T

            MH = np.block([[M11H, M21H],[M12H,M22H]])

            # Solve the system
            z = np.linalg.solve(MH, mulan)
            self.landa = z[:Nnq*Nbodies]
            self.mu = z[Nnq*Nbodies:]
        else:
            M11 = self.M11
            M12 = self.M12
            M21 = self.M21
            M22 = self.M22


            # print('M11 is ', type(M11))
            # print('M11.H[:,0] is', np.conj(M11.T)[:,0])


            Tij = self.Tij

            def M11Hv(v):
                mv = np.zeros(len(v), dtype=complex)
                for ii in range(Nbodies):
                    for jj in range(Nbodies):
                        if not (ii==jj):
                            mv[jj*Nnq:(jj+1)*Nnq] += (
                                np.conj(Tij[ii, jj]) @ (np.conj(self.bodies[ii].B.T) @ v[ii*Nnq:(ii+1)*Nnq]))

                # identity contribution in return
                return -v + mv

            M11H = LinearOperator((Nnq*Nbodies, Nnq*Nbodies), matvec=M11Hv)

            def MHv(v):
                mv = np.zeros(len(v), dtype=complex)
                mv[:Nnq*Nbodies] += M11H@v[:Nnq*Nbodies]
                mv[:Nnq*Nbodies] += np.conj(M21.T)@v[Nnq*Nbodies:]
                mv[Nnq*Nbodies:] += np.conj(M12.T)@v[:Nnq*Nbodies]
                mv[Nnq*Nbodies:] += np.conj(M22.T)@v[Nnq*Nbodies:]

                return mv

            MH = LinearOperator(((Nnq+1)*Nbodies, (Nnq+1)*Nbodies), matvec=MHv)

            # Solve the system
            z, info = gmres(MH, mulan)
            self.landa = np.zeros((Nnq*Nbodies,1), dtype=complex)
            self.landa[:, 0] = z[:Nnq*Nbodies]
            self.mu = np.zeros((Nbodies, 1), dtype=complex)
            self.mu[:, 0] = z[Nnq*Nbodies:]

    #@profile
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
                        W = self.W[ii]

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
            W = self.W[ii]
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
        Nbodies = self.Nbodies
        A = self.scatter_coeffs
        rao = self.rao
        k = self.k
        h1 = self.h1
        h2 = self.h2
        beta = self.beta

        dh = self.h_derivatives()

        dh1_dxi = dh[0]
        dh2_dxi = dh[1]
        dh1_dyi = dh[2]
        dh2_dyi = dh[3]

        landa = self.landa[:, 0]
        mu = self.mu[:, 0]

        dL_dxi = np.zeros(Nbodies)
        dL_dyi = np.zeros(Nbodies)

        if self.denseops:
            dM = self.M_derivatives()

            dM11_dxi = dM[0]
            dM12_dxi = dM[1]
            dM21_dxi = dM[2]
            dM22_dxi = dM[3]

            dM11_dyi = dM[4]
            dM12_dyi = dM[5]
            dM21_dyi = dM[6]
            dM22_dyi = dM[7]

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

        else:
            Nn = self.Nn
            Nq = self.Nq
            Nnq = (2*Nn + 1)*(Nq + 1)

            dT = self.T_derivatives()
            dT_dxi = dT[0]
            dT_dxj = dT[1]
            dT_dyi = dT[2]
            dT_dyj = dT[3]

            gradJx = np.zeros(Nbodies)
            gradJy = np.zeros(Nbodies)

            for kk in range(Nbodies):
                # temporary vectors for hydrodynamics and dynamics
                temp_hyd_x = np.zeros(len(landa), dtype=complex)
                temp_dyn_x = np.zeros(len(mu), dtype=complex)
                temp_hyd_y = np.zeros(len(landa), dtype=complex)
                temp_dyn_y = np.zeros(len(mu), dtype=complex)

                for ii in range(Nbodies):
                    for jj in range(Nbodies):
                        if ii!=jj:
                            B = self.bodies[ii].B
                            R = self.bodies[jj].R
                            Btilde = self.bodies[ii].Btilde
                            Y = self.bodies[ii].Y
                            W = self.W[ii]

                            if kk==ii:
                                ###
                                #dM11_dxi[kk][ii*Nnq:(ii+1)*Nnq, jj*Nnq:(jj+1)*Nnq] = B @ dT_dxj[jj,kk].T
                                temp_hyd_x[ii*Nnq:(ii+1)*Nnq] += B @ (dT_dxj[jj,kk].T @ A[jj*Nnq:(jj+1)*Nnq, 0])

                                #dM12_dxi[kk][ii*Nnq:(ii+1)*Nnq, jj] = (B @ dT_dxj[jj,kk].T @ R)[:,0]
                                temp_hyd_x[ii*Nnq:(ii+1)*Nnq] += (B @ (dT_dxj[jj,kk].T @ R))[:,0] * rao[jj]

                                #dM21_dxi[kk][ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (dT_dxj[jj,kk] @ Btilde.T @ Y).T)[0,:]
                                temp_dyn_x[ii] += (1/W * (dT_dxj[jj,kk] @ (Btilde.T @ Y)).T)[0,:] @ A[jj*Nnq:(jj+1)*Nnq]

                                #dM22_dxi[kk][ii, jj] = 1/W * R.T @ dT_dxj[jj,kk] @ Btilde.T @ Y
                                temp_dyn_x[ii] += 1/W * (R.T @ dT_dxj[jj,kk]) @ (Btilde.T @ Y) * rao[jj]

                                ###
                                #dM11_dyi[kk][ii*Nnq:(ii+1)*Nnq, jj*Nnq:(jj+1)*Nnq] = B @ dT_dyj[jj,kk].T
                                temp_hyd_y[ii*Nnq:(ii+1)*Nnq] += B @ (dT_dyj[jj,kk].T @ A[jj*Nnq:(jj+1)*Nnq, 0])

                                #dM12_dyi[kk][ii*Nnq:(ii+1)*Nnq, jj] = (B @ dT_dyj[jj,kk].T @ R)[:,0]
                                temp_hyd_y[ii*Nnq:(ii+1)*Nnq] += (B @ (dT_dyj[jj,kk].T @ R))[:,0] * rao[jj]

                                #dM21_dyi[kk][ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (dT_dyj[jj,kk] @ Btilde.T @ Y).T)[0,:]
                                temp_dyn_y[ii] += (1/W * (dT_dyj[jj,kk] @ (Btilde.T @ Y)).T)[0,:] @ A[jj*Nnq:(jj+1)*Nnq]

                                #dM22_dyi[kk][ii, jj] = 1/W * R.T @ dT_dyj[jj,kk] @ Btilde.T @ Y
                                temp_dyn_y[ii] += 1/W * (R.T @ dT_dyj[jj,kk]) @ (Btilde.T @ Y) * rao[jj]


                            if kk==jj:
                                ###
                                #dM11_dxi[kk][ii*Nnq:(ii+1)*Nnq, jj*Nnq:(jj+1)*Nnq] = B @ dT_dxi[kk,ii].T
                                temp_hyd_x[ii*Nnq:(ii+1)*Nnq] += B @ (dT_dxi[kk,ii].T @ A[jj*Nnq:(jj+1)*Nnq, 0])

                                #dM12_dxi[kk][ii*Nnq:(ii+1)*Nnq, jj] = (B @ dT_dxi[kk,ii].T @ R)[:,0]
                                temp_hyd_x[ii*Nnq:(ii+1)*Nnq] += (B @ (dT_dxi[kk,ii].T @ R))[:,0] * rao[jj]

                                #dM21_dxi[kk][ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (dT_dxi[kk,ii] @ Btilde.T @ Y).T)[0,:]
                                temp_dyn_x[ii] += (1/W * (dT_dxi[kk,ii] @ (Btilde.T @ Y)).T)[0,:] @ A[jj*Nnq:(jj+1)*Nnq]

                                #dM22_dxi[kk][ii, jj] = 1/W * R.T @ dT_dxi[kk,ii] @ Btilde.T @ Y
                                temp_dyn_x[ii] += 1/W * (R.T @ dT_dxi[kk,ii]) @ (Btilde.T @ Y) * rao[jj]

                                ###
                                #dM11_dyi[kk][ii*Nnq:(ii+1)*Nnq, jj*Nnq:(jj+1)*Nnq] = B @ dT_dyi[kk,ii].T
                                temp_hyd_y[ii*Nnq:(ii+1)*Nnq] += B @ (dT_dyi[kk,ii].T @ A[jj*Nnq:(jj+1)*Nnq, 0])

                                #dM12_dyi[kk][ii*Nnq:(ii+1)*Nnq, jj] = (B @ dT_dyi[kk,ii].T @ R)[:,0]
                                temp_hyd_y[ii*Nnq:(ii+1)*Nnq] += (B @ (dT_dyi[kk,ii].T @ R))[:,0] * rao[jj]

                                #dM21_dyi[kk][ii, jj*Nnq:(jj+1)*Nnq] = (1/W * (dT_dyi[kk,ii] @ Btilde.T @ Y).T)[0,:]
                                temp_dyn_y[ii] += (1/W * (dT_dyi[kk,ii] @ (Btilde.T @ Y)).T)[0,:] @ A[jj*Nnq:(jj+1)*Nnq]

                                #dM22_dyi[kk][ii, jj] = 1/W * R.T @ dT_dyi[kk,ii] @ Btilde.T @ Y
                                temp_dyn_y[ii] += 1/W * (R.T @ dT_dyi[kk,ii]) @ (Btilde.T @ Y) * rao[jj]

                gradJx[kk] = np.real(landa.conj().T @ (temp_hyd_x - dh1_dxi[kk][:,0])
                                + mu.conj().T @ (temp_dyn_x - dh2_dxi[kk][:,0]))
                gradJy[kk] = np.real(landa.conj().T @ (temp_hyd_y - dh1_dyi[kk][:,0])
                                + mu.conj().T @ (temp_dyn_y - dh2_dyi[kk][:,0]))

        return gradJx, gradJy

    def jac_positions(self):
        """
        Computes the Jacobian of the residual Mz-h with respect to
        the positions. Uses sparse operations.
        """
        Nbodies = self.Nbodies
        A = self.scatter_coeffs
        rao = self.rao
        k = self.k
        h1 = self.h1
        h2 = self.h2
        beta = self.beta

        dh = self.h_derivatives()

        dh1_dxi = dh[0]
        dh2_dxi = dh[1]
        dh1_dyi = dh[2]
        dh2_dyi = dh[3]

        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)


        dT = self.T_derivatives()
        dT_dxi = dT[0]
        dT_dxj = dT[1]
        dT_dyi = dT[2]
        dT_dyj = dT[3]

        jac = np.zeros((len(A)+len(rao), 2*Nbodies), dtype=complex)

        for kk in range(Nbodies):
            # temporary vectors for hydrodynamics and dynamics
            temp_hyd_x = np.zeros(len(A), dtype=complex)
            temp_dyn_x = np.zeros(len(rao), dtype=complex)
            temp_hyd_y = np.zeros(len(A), dtype=complex)
            temp_dyn_y = np.zeros(len(rao), dtype=complex)

            for ii in range(Nbodies):
                for jj in range(Nbodies):
                    if ii!=jj:
                        B = self.bodies[ii].B
                        R = self.bodies[jj].R
                        Btilde = self.bodies[ii].Btilde
                        Y = self.bodies[ii].Y
                        W = self.W[ii]

                        if kk==ii:
                            temp_hyd_x[ii*Nnq:(ii+1)*Nnq] += B @ (dT_dxj[jj,kk].T @ A[jj*Nnq:(jj+1)*Nnq, 0])
                            temp_hyd_x[ii*Nnq:(ii+1)*Nnq] += (B @ (dT_dxj[jj,kk].T @ R))[:,0] * rao[jj]
                            temp_dyn_x[ii] += (1/W * (dT_dxj[jj,kk] @ (Btilde.T @ Y)).T)[0,:] @ A[jj*Nnq:(jj+1)*Nnq]
                            temp_dyn_x[ii] += 1/W * (R.T @ dT_dxj[jj,kk]) @ (Btilde.T @ Y) * rao[jj]

                            temp_hyd_y[ii*Nnq:(ii+1)*Nnq] += B @ (dT_dyj[jj,kk].T @ A[jj*Nnq:(jj+1)*Nnq, 0])
                            temp_hyd_y[ii*Nnq:(ii+1)*Nnq] += (B @ (dT_dyj[jj,kk].T @ R))[:,0] * rao[jj]
                            temp_dyn_y[ii] += (1/W * (dT_dyj[jj,kk] @ (Btilde.T @ Y)).T)[0,:] @ A[jj*Nnq:(jj+1)*Nnq]
                            temp_dyn_y[ii] += 1/W * (R.T @ dT_dyj[jj,kk]) @ (Btilde.T @ Y) * rao[jj]


                        if kk==jj:
                            temp_hyd_x[ii*Nnq:(ii+1)*Nnq] += B @ (dT_dxi[kk,ii].T @ A[jj*Nnq:(jj+1)*Nnq, 0])
                            temp_hyd_x[ii*Nnq:(ii+1)*Nnq] += (B @ (dT_dxi[kk,ii].T @ R))[:,0] * rao[jj]
                            temp_dyn_x[ii] += (1/W * (dT_dxi[kk,ii] @ (Btilde.T @ Y)).T)[0,:] @ A[jj*Nnq:(jj+1)*Nnq]
                            temp_dyn_x[ii] += 1/W * (R.T @ dT_dxi[kk,ii]) @ (Btilde.T @ Y) * rao[jj]

                            temp_hyd_y[ii*Nnq:(ii+1)*Nnq] += B @ (dT_dyi[kk,ii].T @ A[jj*Nnq:(jj+1)*Nnq, 0])
                            temp_hyd_y[ii*Nnq:(ii+1)*Nnq] += (B @ (dT_dyi[kk,ii].T @ R))[:,0] * rao[jj]
                            temp_dyn_y[ii] += (1/W * (dT_dyi[kk,ii] @ (Btilde.T @ Y)).T)[0,:] @ A[jj*Nnq:(jj+1)*Nnq]
                            temp_dyn_y[ii] += 1/W * (R.T @ dT_dyi[kk,ii]) @ (Btilde.T @ Y) * rao[jj]

            jac[:len(A),kk] = temp_hyd_x - dh1_dxi[kk][:,0]
            jac[len(A):,kk] = temp_dyn_x - dh2_dxi[kk][:,0]
            jac[:len(A),kk+Nbodies] = temp_hyd_y - dh1_dyi[kk][:,0]
            jac[len(A):,kk+Nbodies] = temp_dyn_y - dh2_dyi[kk][:,0]

        return jac



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
            W = self.W[ii]
            Btilde = self.bodies[ii].Btilde
            Y = self.bodies[ii].Y

            for jj in range(Nbodies): # loop on columns
                if jj!=ii:
                    Tij = self.Tij[ii, jj]

                    R = self.bodies[jj].R
                    dM21_dci[ii][ii, jj*Nnq:(jj+1)*Nnq] = -1/(W**2)*omega/(rho*g)*((Tij@Btilde.T@Y).T)
                    dM22_dci[ii][ii, jj] = -1/(W**2)*(R.T@Tij@Btilde.T@Y)*omega/(rho*g)

        return dM21_dci, dM22_dci


    def jac_imped(self):
        """
        Computes the jacobian of the residual of the dynamic equation with respect
        to the impedance. Used to build the reduced gradient with respect
        to the control parameters with sparse operations.
        Added on 01/09/2023. TODO: document.
        """
        g = self.g
        rho = self.water_density
        k = self.k
        beta = self.beta
        Nn = self.Nn
        Nq = self.Nq
        Nnq = (2*Nn + 1)*(Nq + 1)
        omega = self.omega
        Nbodies = self.Nbodies

        rao = self.rao[:, 0]
        A = self.scatter_coeffs[:, 0]

        # The Jacobian is diagonal, only the diagonal is stored
        jac = np.zeros(Nbodies, dtype=complex)
        for ii in range(Nbodies):
            W = self.W[ii]
            Btilde = self.bodies[ii].Btilde
            Y = self.bodies[ii].Y
            a = self.bodies[ii].radius
            inc_wave_coeffs = self.incident_wave_coeffs(k, beta, a, self.x[ii], self.y[ii], Nn, Nq)

            for jj in range(Nbodies): # loop on columns
                if jj!=ii:
                    Tij = self.Tij[ii, jj]
                    R = self.bodies[jj].R

                    jac[ii] -= Y.T@Btilde@Tij.T@A[jj*Nnq:(jj+1)*Nnq]
                    jac[ii] -= (R.T@Tij@Btilde.T@Y)*rao[jj]

            # rhs contribution
            jac[ii] -= inc_wave_coeffs.T@Btilde.T@Y
            # common factor
            jac[ii] *= 1/W**2

        return jac


    def gradientJ_dampstiff(self):
        """
        Computes the gradient of the Lagrangian with respect to the
        damping and stiffness coefficients.
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

        DP = np.zeros(Nbodies)

        for ii in range(Nbodies):
            DP[ii] = -0.5*omega**2 * np.abs(rao[ii])**2

        jac = self.jac_imped()
        dL_dci = DP + omega/(rho*g) * np.real(mu.conj().T*jac)
        dL_dsi = np.real(1j/(rho*g)*mu.conj().T*jac)

        return dL_dci[0, :], dL_dsi[0, :]
