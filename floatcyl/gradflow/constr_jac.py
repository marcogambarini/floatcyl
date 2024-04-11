#! /usr/bin/env python3

"""
Computation of the constraint function and Jacobian matrix
"""

import numpy as np
import firedrake as fd
from scipy.sparse import csr_array
from .inoutpolygonmesh import PolygonHole

class PolyConstraint:
    def __init__(self, vertices, meshFileName='rectpoly',
                 rect_factor=1.5, h_outer=0.2, h_inner=0.05,
                 save_pvd=False, conds='dirneum',
                 projmethod='CG', nu=0):
        """
        Solves the internal-external Poisson problem to compute the
        constraint function for the polygon

        Parameters
        ----------
        vertices: array of shape (Nv, 2)
            Ordered vertices of the polygon (both clockwise and
            counterclockwise are ok)
        meshFileName: string
            Default: 'rectpoly'
        rect_factor: float
            Relative size of bounding rectangle wrt to the polygon
            (default: 1.5)
        h_outer: float
            Mesh size at outer rectangle boundary (default: 0.2)
        h_inner: float
            Mesh size at inner polygon boundary (default: 0.05)
        save_pvd: boolean
            Whether to save the PDE solution as a pvd file (default: False)
        conds: string
            'dirneum' (default) for Dirichlet on the polygon, Neumann
            on the rectangle; 'dirdir' for Dirichlet on both
        projmethod: string
            Method for gradient recovery. 'CG' (default): projection on a space
            of continuous linear functions. 'DG': projection on a space of
            discontinuous, elementwise constant functions. 'DG' yields
            the exact gradient.
        nu: float
            Constant for H1 regularization of the gradient (default: 0)
        """

        # Define a rectangle containing the polygon
        bbox_barycenter = np.array([0.5*(np.min(vertices[:,0])+np.max(vertices[:,0])),
                                    0.5*(np.min(vertices[:,1])+np.max(vertices[:,1]))])
        Lx, Ly = rect_factor * np.max(
                    np.abs(vertices-bbox_barycenter), axis=0)
        rect_vertices = bbox_barycenter + np.array(((-Lx, -Ly),
                                               (-Lx,  Ly),
                                               ( Lx,  Ly),
                                               ( Lx, -Ly)))

        self.Lx, self.Ly = Lx, Ly

        # Build the mesh
        domain = PolygonHole(rect_vertices, vertices,
                             h_outer=h_outer, h_inner=h_inner)
        domain.mesh(meshFileName, show=False, vtk=True)

        self.meshFileName = meshFileName

        self.conds = conds
        self.projmethod = projmethod
        self.save_pvd = save_pvd
        self.nu = nu
        self.solve_fem_post()




    def solve_fem_post(self):
        """
        Solve the Poisson equation in its standard weak form and
        then recover the gradient as post-processing projection
        """

        # Solve the Poisson problem using Firedrake
        mesh = fd.Mesh(self.meshFileName + '.msh')
        # Boundary tags: 1 farfield, 2 polygon
        # Domain tags: 3 outer domain, 4 inner domain

        #################### Poisson problem ##########################
        V = fd.FunctionSpace(mesh, 'P', 1)
        projdeg = 1

        if self.conds=='dirdir':
            outer_gD = fd.Constant(max(self.Lx, self.Ly))
            bc = [fd.DirichletBC(V, fd.Constant(0.), (2)),
                fd.DirichletBC(V, outer_gD, (1))]
            f_out = fd.Constant(0.)
        elif self.conds=='dirneum':
            bc = fd.DirichletBC(V, fd.Constant(0.), (2))
            f_out = fd.Constant(1.)
        else:
            raise ValueError('conds can be either dirdir or dirneum')

        f_int = fd.Constant(-1.)

        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)

        a = fd.dot(fd.grad(u), fd.grad(v)) * fd.dx
        L = f_out * v * fd.dx(3) + f_int * v * fd.dx(4)

        u_h = fd.Function(V)
        fd.solve(a==L, u_h, bcs=bc)

        ######## gradient recovery: L2/H1 projection #########
        if self.projmethod=='CG':
            W = fd.VectorFunctionSpace(mesh, 'P', projdeg)
        elif self.projmethod=='DG':
            W = fd.VectorFunctionSpace(mesh, 'DG', projdeg - 1)
        else:
            raise ValueError('projmethod can be either CG or DG')

        p = fd.TrialFunction(W)
        q = fd.TestFunction(W)
        a_proj = (fd.Constant(self.nu)*fd.inner(fd.grad(p), fd.grad(q)) *fd.dx
                  + fd.dot(p, q) * fd.dx)
        L_proj = fd.dot(fd.grad(u_h), q) * fd.dx
        p_h = fd.Function(W)
        fd.solve(a_proj==L_proj, p_h)


        if self.save_pvd:
            outfile = fd.File(self.meshFileName + '.pvd')
            outfile.write(u_h, p_h)

        # Keep the solution as an attribute of the class
        self.u = u_h
        self.gradu = p_h


    def compute_g(self, x, y, s, d, compute_jac=True):
        """
        Computes the values and optionally the Jacobian matrix of the
        constraint functions

        Parameters
        ----------
        x, y: arrays
            coordinates of points
        s: array
            slack variable vector
        d: float
            minimum distance between points
        compute_jac: boolean
            whether to compute also the Jacobian (default: True)
        """
        Nb = len(x)
        Nc = Nb*(Nb-1)//2 + Nb # total number of constraints
        g = np.zeros(Nc)
        if s is None:
            s = np.zeros(Nc)
        kg = 0 # counter for elements of g
        if compute_jac:
            Ns = 2 * Nb * Nb # total number of nonzero elements of J
            indptr = np.zeros(Nc+1, dtype=int)
            indices = np.zeros(Ns, dtype=int)
            data = np.zeros(Ns, dtype=float)
            indptr[-1] = Ns
            kJ = 0 # counter for elements of data and indices

        # Minimum distance constraints
        for ii in range(Nb):
            for jj in range(ii+1, Nb):
                g[kg] = d*d - ((x[ii] - x[jj])**2 + (y[ii] - y[jj])**2) + s[kg]*s[kg]
                if compute_jac:
                    indptr[kg] = kJ
                    indices[kJ:kJ+4] = [ii, jj, ii+Nb, jj+Nb]
                    data[kJ:kJ+4] = ([-2*(x[ii] - x[jj]),
                                    -2*(x[jj] - x[ii]),
                                    -2*(y[ii] - y[jj]),
                                    -2*(y[jj] - y[ii])])
                    kJ += 4
                kg += 1

        # Domain constraints
        for ii in range(Nb):
            xii_vec = np.array((x[ii], y[ii]))
            g[kg] = (self.u.at(xii_vec) + s[kg]*s[kg])
            if compute_jac:
                indptr[kg] = kJ
                indices[kJ:kJ+2] = [ii, ii+Nb]
                data[kJ:kJ+2] = self.gradu.at(xii_vec)
                kJ += 2
            kg += 1

        if compute_jac:
            J = csr_array((data, indices, indptr), shape=(Nc, 2*Nb))
            return g, J
        else:
            return g


    def plot_field(self):
        """
        Plots the solution of the PDE using firedrake's utilities
        """
        import matplotlib.pyplot as plt
        import firedrake.pyplot as fdplt
        u_h = self.u
        p_h = self.gradu

        fig, ax = plt.subplots()
        q = fdplt.tripcolor(u_h, axes=ax)
        #fdplt.tricontour(u_h, axes=ax, levels=[0.], colors='red')
        #fdplt.tricontour(u_h, axes=ax)
        fdplt.streamplot(p_h, axes=ax, linewidth=2, cmap='autumn',
                         density=10.)
        ax.set_aspect('equal')
        plt.colorbar(q)

        plt.show()
