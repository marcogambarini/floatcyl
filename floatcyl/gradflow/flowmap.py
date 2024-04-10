#! /usr/bin/env python3

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg, LinearOperator
from time import perf_counter

class linsys_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1

class Flowmap(object):
    def __init__(self, cylArrays, domain_constr, alpha_slam, min_dist,
            adapt_tol=False, cg_tol=1e-5):
        """
        Parameters
        ----------
        cylArrays: list of floatcyl Array objects
            A floatcyl Array for each frequency of the discretized spectrum
        domain_constr: PolyConstraint object (from constr_jac)
            Class containing methods to compute domain constraints
        alpha_slam: float
            Constant for slamming constraint
        min_dist: float
            Minimum distance between objects
        adapt_tol: boolean
            Whether to adapt the tolerance of CG to the current value
            of norm(Phi)
        cg_tol: float
            Tolerance for CG. Ignored if adapt_tol is True
        """

        self.cylArrays = cylArrays
        self.domain_constr = domain_constr

        self.Nf = len(cylArrays)

        # Obtain discretization data from first Array
        self.Nbodies = cylArrays[0].Nbodies
        self.Nn = cylArrays[0].Nn
        self.Nq = cylArrays[0].Nq
        self.Nnq = (2*self.Nn + 1)*(self.Nq + 1) # number of hydrodynamic unknowns

        # Compute number of constraints
        self.Ncx = self.Nbodies*(self.Nbodies-1)//2 + self.Nbodies
        self.Ncz = self.Nbodies

        # Initialize linear system initial guess
        self.oldSol = None
        self.oldRestoreSol = None

        # Slamming constraint constant
        self.alpha_slam = alpha_slam

        # Minimum distance between objects
        self.min_dist = min_dist

        # Build a vector with drafts of all objects
        # This allows using arrays with objects of different sizes
        self.drafts = np.array(
            [cylArrays[0].bodies[ii].draft for ii in range(self.Nbodies)])

        # Initialize cg tolerance
        self.cg_tol = None
        self.adapt_tol = adapt_tol
        if adapt_tol:
            # Initialization ofstimate of the pseudoinverse norm and Phi norm
            self.pinvJgnorm = 1.
            self.Phinorm = 1.
        else:
            self.cg_tol = cg_tol



    def initial_state_and_global_scalings(self, x0, y0, gen_damping, gen_stiffness):
        """
        Computes the initial values of the state variables (cylArrays),
        initializes the slack variables and defines global scalings
        (cost and controls)
        damping and stiffness must be initialized to nonzero values (zero power
        is a stationary point)

        Parameters
        ----------
        x0, y0: arrays
            Initial guess for coordinates of points
        gen_damping: float
            Initial guess for generator damping
        gen_stiffness: float
            Initial guess for generator stiffness

        Returns
        -------
        W0: complex array
            Full nondimensional initial guess vector
        """
        Ncx, Nf, Nbodies = self.Ncx, self.Nf, self.Nbodies
        cylArrays, domain_constr = self.cylArrays, self.domain_constr
        alpha_slam, min_dist = self.alpha_slam, self.min_dist

        self.x_scale = max(np.max(np.abs(x0)), np.max(np.abs(y0)))

        # Initial guess for the position slack variables
        gx = domain_constr.compute_g(x0, y0, np.zeros(Ncx), min_dist,
                compute_jac=False)
        s0x_overlap = np.sqrt(np.abs(gx[:-Nbodies]))
        self.sx_overlap_scale = max(1., np.max(s0x_overlap))
        s0x_domain = np.sqrt(np.abs(gx[-Nbodies:]))
        self.sx_domain_scale = max(1., np.max(s0x_domain))

        z0_vec = []
        gs = np.zeros(Nbodies)
        cost_scale = 0
        for ii in range(Nf):
            cylArray = cylArrays[ii]
            # Initial guess for the state variables: solution of the state problem
            # at the initial position
            etavec = 1j * cylArray.H/2 * np.exp(1j*cylArray.k*(
                                np.cos(cylArray.beta) * x0
                                + np.sin(cylArray.beta) * y0
                            ))
            cylArray.solve()
            cost_scale += cylArray.compute_power()
            z0_vec.append(np.block([cylArray.scatter_coeffs[:,0], cylArray.rao[:,0]]))

            # Slamming constraint residual
            slam = z0_vec[ii][-Nbodies:] - etavec
            gs += np.abs(slam)**2

        self.z_scale = np.max([np.linalg.norm(zz) for zz in z0_vec])
        #self.z_scale = 1.
        self.cost_scale = cost_scale

        gs = gs - 2*(alpha_slam*self.drafts)**2
        if all(gs<0):
            s0z = np.sqrt(-gs)
        else:
            s0z = np.ones(Nbodies)
        self.sz_scale = max(1., np.sqrt(np.max(-gs)))

        self.damp_scale = self.stiff_scale = max(abs(gen_damping),
                                                 abs(gen_stiffness))
        #print('scalings = ', self.x_scale, self.z_scale, self.damp_scale,
        #   self.stiff_scale, self.sx_overlap_scale, self.sx_domain_scale,
        #   self.sz_scale)

        # self.x_scale =  25.0
        # self.damp_scale =  100000.0
        # self.stiff_scale =  100000.0
        # self.sx_overlap_scale =  49.109813410985964
        # self.sx_domain_scale =  49.109813410985964
        # self.sz_scale =  1.0
        # self.z_scale = 1.2


        # Scaled initial vector
        W0 = np.block([x0/self.x_scale, y0/self.x_scale,
                        np.concatenate([z0_vec[ii] for ii in range(Nf)])/self.z_scale,
                        gen_damping*np.ones(Nbodies)/self.damp_scale,
                        gen_stiffness*np.ones(Nbodies)/self.stiff_scale,
                        s0x_overlap/self.sx_overlap_scale,
                        s0x_domain/self.sx_domain_scale,
                        s0z/self.sz_scale])

        return W0


    def w_split(self, w):
        """
        Splits the nondimensional vector of controls and slacks and makes it dimensional

        Parameters
        ----------
        w: complex array
            Nondimensional control vector

        Returns
        -------
        x, y, zvec, c, k, sx_overlap, sx_domain, sz: arrays
            Slices of the control vector, made dimensional
        """
        Nbodies, Nnq, Ncx, Nf = self.Nbodies, self.Nnq, self.Ncx, self.Nf

        x = np.real(w[:Nbodies])
        y = np.real(w[Nbodies:2*Nbodies])
        row_index = 2*Nbodies
        zvec = []
        for ii in range(Nf):
            zvec.append(w[row_index:row_index+(Nnq+1)*Nbodies])
            row_index += (Nnq+1)*Nbodies
        c = np.real(w[row_index:row_index+Nbodies])
        k = np.real(w[row_index+Nbodies:row_index+2*Nbodies])
        # slack variables for position constraints
        sx_overlap = np.real(w[row_index+2*Nbodies:row_index+2*Nbodies+Ncx-Nbodies])
        sx_domain = np.real(w[row_index+2*Nbodies+Ncx-Nbodies:row_index+2*Nbodies+Ncx])
        # slack variables for the slamming constraints
        sz = np.real(w[row_index+2*Nbodies+Ncx:])

        # Scale back the variables
        x = x * self.x_scale
        y = y * self.x_scale
        zvec = [zz * self.z_scale for zz in zvec]
        c = c * self.damp_scale
        k = k * self.stiff_scale
        sx_overlap = sx_overlap * self.sx_overlap_scale
        sx_domain = sx_domain * self.sx_domain_scale
        sz = sz * self.sz_scale

        return x, y, zvec, c, k, sx_overlap, sx_domain, sz


    def compute_f_g(self, w):
        """
        Computes cost and constraint functions with constant scaling
        (so that they can be compared between iterations)

        Parameters
        ----------
        w: complex array
            Nondimensional control vector

        Returns
        -------
        f: float
            Nondimensional cost
        g: array
            Vector of nondimensional constraint functions
        """
        x, y, zvec, c, k, sx_overlap, sx_domain, sz = self.w_split(w)
        cylArrays, domain_constr = self.cylArrays, self.domain_constr
        Nf, Nbodies, Nnq = self.Nf, self.Nbodies, self.Nnq
        min_dist, alpha_slam = self.min_dist, self.alpha_slam


        # Update the arrays
        for ii in range(Nf):
            cylArrays[ii].x = x
            cylArrays[ii].y = y
            cylArrays[ii].update_controls(c, k)
            # Update scatter coefficients and rao
            # Since no explicit solve is performed, these come from the time-stepping
            cylArrays[ii].scatter_coeffs = zvec[ii][:-Nbodies].reshape((Nnq*Nbodies, 1))
            cylArrays[ii].rao = zvec[ii][-Nbodies:].reshape((Nbodies, 1))
            cylArrays[ii].basis_transformation_matrices()


        power = np.sum([cylArray.compute_power() for cylArray in cylArrays])
        print('power = ', power)

        # Compute position constraint functions
        gx = domain_constr.compute_g(
            x, y, np.concatenate([sx_overlap, sx_domain]),
            min_dist, compute_jac=False)

        res_vec = []
        gs = np.zeros(Nbodies)
        for ii in range(Nf):
            cylArray = cylArrays[ii]
            # Compute state problem residual
            M, _, hh = cylArray.expose_operators(returnblocks=False)
            res_vec.append(M@zvec[ii] - hh)

            # Compute values and gradients of the slamming constraint
            # Incident wave height
            etavec = 1j * cylArray.H/2 * np.exp(1j*cylArray.k*(
                                np.cos(cylArray.beta) * x
                                + np.sin(cylArray.beta) * y
                            ))
            slam = zvec[ii][-Nbodies:] - etavec
            # Slamming constraint update
            gs += np.abs(slam)**2

        #print('slamming amplitudes = ', np.sqrt(gs))

        gs = gs - 2*(alpha_slam*self.drafts)**2 + sz**2

        rhs8re = np.concatenate([
            np.real(res_vec[ii])
            for ii in range(Nf)
        ])

        rhs8im = np.concatenate([
            np.imag(res_vec[ii])
            for ii in range(Nf)
        ])

        # Residuals are not scaled: we know how small we want them (makes sense?)
        g_vec = np.concatenate([gx/self.x_scale**2,
                        rhs8re,
                        rhs8im,
                        gs/np.max(self.drafts)**2])

        return (-power/self.cost_scale, g_vec)


    def compute_phi(self, w, alpha_F=1, alpha_G=1, monitor=False):
        """
        Computes cost and constraint functions with constant scaling
        (so that they can be compared between iterations), same as in compute_f_g,
        and the gradient flow direction Phi

        alpha_F, alpha_G are the constants alpha_J, alpha_C of Feppon, respectively
        setting alpha_F = 0 starts a feasibility restoration phase

        Parameters
        ----------
        w: complex float array
            Control vector
        alpha_F: float
            Weight of the projected gradient contribution (default: 1.)
        alpha_G: float
            Weight of the constraint restoration constribution (default: 1.)
        monitor: boolean
            Whether to save performance monitoring data (default: False)

        Returns
        -------
        f: float
            Nondimensional cost function
        g: array
            Vector of nondimensional constraint functions
        Phi: array
            Gradient flow direction
        """
        if monitor:
            monitor_t0 = perf_counter()

        x, y, zvec, c, k, sx_overlap, sx_domain, sz = self.w_split(w)
        sx = np.concatenate([sx_overlap, sx_domain])
        cylArrays, domain_constr = self.cylArrays, self.domain_constr
        Nf, Nbodies, Nnq, Ncx, Ncz = self.Nf, self.Nbodies, self.Nnq, self.Ncx, self.Ncz
        min_dist, alpha_slam = self.min_dist, self.alpha_slam
        oldSol, oldRestoreSol = self.oldSol, self.oldRestoreSol


        # Update the arrays
        for ii in range(Nf):
            cylArrays[ii].x = x
            cylArrays[ii].y = y
            cylArrays[ii].update_controls(c, k)
            # Update scatter coefficients and rao
            # Since no explicit solve is performed, these come from the time-stepping
            cylArrays[ii].scatter_coeffs = zvec[ii][:-Nbodies].reshape((Nnq*Nbodies, 1))
            cylArrays[ii].rao = zvec[ii][-Nbodies:].reshape((Nbodies, 1))
            cylArrays[ii].basis_transformation_matrices()

        power = np.sum([cylArray.compute_power() for cylArray in cylArrays])
        print('power = ', power)

        # Compute position constraint functions and Jacobian
        gx, Jg_xx = domain_constr.compute_g(
            x, y, sx, min_dist)
        Jg_xs = 2*sx # diagonal matrix, represented as a vector

        M_vec = []
        MH_vec = []
        Jg_zx_vec = []
        Jg_c_vec = []
        Jg_k_vec = []
        res_vec = []
        slam_mat = np.zeros((Nf, Nbodies), dtype=complex)
        eta_mat = np.zeros((Nf, Nbodies), dtype=complex)
        gs = np.zeros(Nbodies)
        for ii in range(Nf):
            cylArray = cylArrays[ii]
            rho = cylArray.water_density
            g = cylArray.g
            # Compute state problem residual
            M, MH, hh = cylArray.expose_operators(returnblocks=False)
            res_vec.append(M@zvec[ii] - hh)
            #print('norm of residual = ', np.linalg.norm(res_vec[ii]))
            # Jacobian of the residual with respect to positions
            Jg_zx = cylArray.jac_positions()
            # Jacobian of the residual with respect to impedance
            Jg_imp = cylArray.jac_imped()
            Jg_c = cylArray.omega/(rho*g) * Jg_imp  # damping
            Jg_k = 1j/(rho*g) * Jg_imp  # stiffness
            # Compute values and gradients of the slamming constraint
            # Incident wave height
            etavec = 1j * cylArray.H/2 * np.exp(1j*cylArray.k*(
                                np.cos(cylArray.beta) * x
                                + np.sin(cylArray.beta) * y
                            ))
            slam = zvec[ii][-Nbodies:] - etavec
            # Slamming constraint update
            gs += np.abs(slam)**2

            M_vec.append(M)
            MH_vec.append(MH)
            Jg_zx_vec.append(Jg_zx)
            Jg_c_vec.append(Jg_c)
            Jg_k_vec.append(Jg_k)
            slam_mat[ii, :] = slam
            eta_mat[ii, :] = etavec

        #print('slamming amplitudes = ', np.sqrt(gs))

        # diagonals of Jacobian wrt positions
        Jg_sx = np.zeros(2*Nbodies)
        # x components
        Jg_sx[:Nbodies] = np.sum([2*cylArrays[ii].k*np.cos(cylArrays[ii].beta) *
                            np.real(1j*np.conj(eta_mat[ii])*zvec[ii][-Nbodies:])
                            for ii in range(Nf)], axis=0)
        # y components
        Jg_sx[Nbodies:] = np.sum([2*cylArrays[ii].k*np.sin(cylArrays[ii].beta) *
                            np.real(1j*np.conj(eta_mat[ii])*zvec[ii][-Nbodies:])
                            for ii in range(Nf)], axis=0)
        Jg_ss = 2*sz


        gs = gs - 2*(alpha_slam*self.drafts)**2 + sz**2
        #print('norm of slamming constraint = ', np.linalg.norm(gs))

        # Rescale Jacobians according to the scales of states
        Jg_xx = Jg_xx * self.x_scale
        Jg_xs = np.concatenate([Jg_xs[:-Nbodies] * self.sx_overlap_scale,
                                Jg_xs[-Nbodies:] * self.sx_domain_scale])
        for ii in range(Nf):
            Jg_zx_vec[ii] = Jg_zx_vec[ii] * self.x_scale
            Jg_c_vec[ii] = Jg_c_vec[ii] * self.damp_scale
            Jg_k_vec[ii] = Jg_k_vec[ii] * self.stiff_scale
        Jg_sx = Jg_sx * self.x_scale
        Jg_ss = Jg_ss * self.sz_scale


        # Rescale Jacobians and constraints
        # Improves the condition number of the algebraic problem
        # Does not influence the time evolution

        # scale all rows separately
        v = np.ones(Nbodies*2)
        xconstrscales = np.sqrt((Jg_xx)**2@v + Jg_xs**2)
        xconstrscaling = diags(1/xconstrscales)

        #w = np.zeros((Nnq+1)*Nbodies)
        #w[0] = 1
        #zconstrscale = np.sqrt(np.linalg.norm(M@w)**2 + np.linalg.norm(Jg_zx@v)**2)
        zconstrscale = 1

        sconstrscale = np.sqrt(Jg_sx[0]**2 + 4*np.abs(slam_mat[0,0])**2 + Jg_ss[0]**2)

        #print('constraint scales = ', xconstrscales, zconstrscale, sconstrscale)

        Jg_xx = xconstrscaling@Jg_xx
        Jg_xs = xconstrscaling@Jg_xs
        for ii in range(Nf):
            Jg_zx_vec[ii] = Jg_zx_vec[ii] / zconstrscale
            Jg_c_vec[ii] = Jg_c_vec[ii] / zconstrscale
            Jg_k_vec[ii] = Jg_k_vec[ii] / zconstrscale
        Jg_sx = Jg_sx / sconstrscale
        Jg_ss = Jg_ss / sconstrscale

        def mvre_Jg(v):
            # slice into components
            Phi_x = v[:2*Nbodies]
            row_index_re = 2*Nbodies
            row_index_im = 2*Nbodies + Nf*(Nnq+1)*Nbodies
            # order of Phi_z variables in v: first all real parts, then all imag parts
            Phi_z_vec = []
            for ii in range(Nf):
                Phi_z_vec.append(v[row_index_re:row_index_re+(Nnq+1)*Nbodies] + 0*1j)
                row_index_re += (Nnq+1)*Nbodies
                Phi_z_vec[ii] += 1j*v[row_index_im:row_index_im+(Nnq+1)*Nbodies]
                row_index_im += (Nnq+1)*Nbodies
            row_index = row_index_im
            Phi_c = v[row_index:row_index+Nbodies]
            row_index += Nbodies
            Phi_k = v[row_index:row_index+Nbodies]
            row_index += Nbodies
            Phi_sx = v[row_index:row_index+Ncx]
            row_index += Ncx
            Phi_sz = v[row_index:row_index+Ncz]

            mv1 = Jg_xx@Phi_x + Jg_xs*Phi_sx
            mv2 = np.zeros(Nf*(Nnq+1)*Nbodies, dtype=complex)
            row_index = 0
            for ii in range(Nf):
                mv2ii = Jg_zx_vec[ii]@Phi_x + self.z_scale*(M_vec[ii]@Phi_z_vec[ii])/zconstrscale
                mv2ii[-Nbodies:] += Jg_c_vec[ii]*Phi_c + Jg_k_vec[ii]*Phi_k
                mv2[row_index:row_index+(Nnq+1)*Nbodies] = mv2ii
                row_index += (Nnq+1)*Nbodies
            mv3 = (Jg_sx[:Nbodies]*Phi_x[:Nbodies] + Jg_sx[Nbodies:]*Phi_x[Nbodies:]
                + 2*self.z_scale*np.sum([np.real(np.conj(slam_mat[ii])*Phi_z_vec[ii][-Nbodies:])/sconstrscale
                            for ii in range(Nf)], axis=0)
                + Jg_ss*Phi_sz)

            return np.concatenate([mv1, np.real(mv2), np.imag(mv2), mv3])


        def mvre_JgT(v):
            # slice into components
            row_index = 0
            Lam_x = v[row_index:row_index+Ncx]
            row_index += Ncx
            # order of Lam_z variables in v: first all real parts, then all imag parts
            Lam_z_vec = []
            row_index_re = row_index
            row_index_im = row_index + Nf*(Nnq+1)*Nbodies
            for ii in range(Nf):
                Lam_z_vec.append(v[row_index_re:row_index_re+(Nnq+1)*Nbodies] + 0*1j)
                row_index_re += (Nnq+1)*Nbodies
                Lam_z_vec[ii] += 1j*v[row_index_im:row_index_im+(Nnq+1)*Nbodies]
                row_index_im += (Nnq+1)*Nbodies
            row_index = row_index_im
            Lam_s = v[row_index:]

            mv1 = (Jg_xx.T@Lam_x +
                    np.sum([np.real(np.conj(Jg_zx_vec[ii].T)@Lam_z_vec[ii]) for ii in range(Nf)],
                            axis=0))
            mv1[:Nbodies] += Jg_sx[:Nbodies] * Lam_s
            mv1[Nbodies:] += Jg_sx[Nbodies:] * Lam_s
            mv2 = np.zeros(Nf*(Nnq+1)*Nbodies, dtype=complex)
            row_index = 0
            for ii in range(Nf):
                mv2ii = self.z_scale*(MH_vec[ii]@Lam_z_vec[ii])/zconstrscale
                mv2ii[-Nbodies:] += 2*self.z_scale*slam_mat[ii]*Lam_s/sconstrscale
                mv2[row_index:row_index+(Nnq+1)*Nbodies] = mv2ii
                row_index += (Nnq+1)*Nbodies
            mv3 = np.sum([np.real(np.conj(Jg_c_vec[ii])*Lam_z_vec[ii][-Nbodies:])
                                    for ii in range(Nf)],
                                    axis=0)
            mv4 = np.sum([np.real(np.conj(Jg_k_vec[ii])*Lam_z_vec[ii][-Nbodies:])
                                    for ii in range(Nf)],
                                    axis=0)
            mv5 = Jg_xs*Lam_x
            mv6 = Jg_ss*Lam_s

            return np.concatenate([mv1, np.real(mv2), np.imag(mv2),
                            mv3, mv4, mv5, mv6])

        ########################## CG ##############################
        def mv_re(v):
            if v.ndim>1:
                v = v[:,0]
            return mvre_Jg(mvre_JgT(v))

        Na = Nf*(Nnq+1)*Nbodies + Ncx + Ncz
        Na_re = Na + Nf*(1+Nnq)*Nbodies
        A = LinearOperator((Na_re, Na_re), matvec=mv_re)


        rhs2re = np.concatenate([
            np.concatenate([np.zeros(Nnq*Nbodies),
            self.z_scale/self.cost_scale * cylArrays[ii].omega**2 * c*np.real(zvec[ii][-Nbodies:])])
            for ii in range(Nf)
        ])

        rhs2im = np.concatenate([
            np.concatenate([np.zeros(Nnq*Nbodies),
            self.z_scale/self.cost_scale * cylArrays[ii].omega**2 * c*np.imag(zvec[ii][-Nbodies:])])
            for ii in range(Nf)
        ])

        rhs8re = np.concatenate([
            -np.real(res_vec[ii])/zconstrscale
            for ii in range(Nf)
        ])

        rhs8im = np.concatenate([
            -np.imag(res_vec[ii])/zconstrscale
            for ii in range(Nf)
        ])

        gradF = -np.concatenate([np.zeros(2*Nbodies),
                        rhs2re,
                        rhs2im,
                        self.damp_scale/self.cost_scale * np.sum(
                                [cylArrays[ii].omega**2 * 0.5*np.abs(zvec[ii][-Nbodies:])**2
                                        for ii in range(Nf)], axis=0),
                        np.zeros(Nbodies),
                        np.zeros(Ncx),
                        np.zeros(Ncz)])
        G = -np.concatenate([-xconstrscaling@gx,
                        rhs8re,
                        rhs8im,
                        -gs/sconstrscale])
        rhs = alpha_G*G - alpha_F*mvre_Jg(gradF)

        #print('rhsnorm = ', np.linalg.norm(rhs))

        #print('zscale = ', self.z_scale)
        #print(np.linalg.norm(gradF))
        #print(np.linalg.norm(G))
        counter = linsys_counter()

        if self.adapt_tol:
            self.cg_tol = min(0.1*np.linalg.norm(G), 0.1*self.Phinorm/self.pinvJgnorm)
            print('Estimated required CG tolerance = ', self.cg_tol)

        # solve
        if monitor:
            monitor_t0_CG = perf_counter()
        if alpha_F==0:
            lam, info = cg(A, rhs, x0=oldRestoreSol, callback=counter, maxiter=2000)
            print(counter.niter, ' CG iterations')
            self.oldRestoreSol = lam
        else:
            lam, info = cg(A, rhs, x0=oldSol, callback=counter, tol=self.cg_tol, maxiter=2000)
            print(counter.niter, ' CG iterations')
            self.oldSol = lam
        if monitor:
            monitor_time_CG = perf_counter() - monitor_t0_CG

        if info==0:
            print('successful CG iterations')


        ulam = mvre_JgT(lam)
        u = -alpha_F*gradF - ulam
        Phi = np.concatenate([u[:2*Nbodies],
                    (u[2*Nbodies:2*Nbodies+Nf*(Nnq+1)*Nbodies]
                    + 1j* u[2*Nbodies+Nf*(Nnq+1)*Nbodies:2*Nbodies+2*Nf*(Nnq+1)*Nbodies]),
                    u[2*Nbodies+2*Nf*(Nnq+1)*Nbodies:4*Nbodies+2*Nf*(Nnq+1)*Nbodies+Ncx+Ncz]])

        if self.adapt_tol:
            self.pinvJgnorm = np.linalg.norm(ulam)/np.linalg.norm(rhs)
            self.Phinorm = np.linalg.norm(Phi)

        if monitor:
            monitor_time = perf_counter() - monitor_t0

            return (-power/self.cost_scale,
                    np.concatenate([gx/self.x_scale**2,
                            rhs8re,
                            rhs8im,
                            gs/np.max(self.drafts)**2]),
                    Phi, counter.niter, monitor_time, monitor_time_CG)
        else:
            # returns f, g, Phi
            return (-power/self.cost_scale,
                    np.concatenate([gx/self.x_scale**2,
                            rhs8re,
                            rhs8im,
                            gs/np.max(self.drafts)**2]),
                    Phi)
