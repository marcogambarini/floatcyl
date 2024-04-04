#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class PolyProjector:
    def __init__(self, vertices):
        """
        Constructs the projection matrix on each edge

        Parameters
        ----------
        vertices: list of 2D arrays
            Vertices of the polygon in clockwise order
        """
        N = len(vertices)

        dlist = []
        Mlist = []
        nlist = []

        for ii in range(N):
            d = vertices[ii] - vertices[ii-1] # direction vector
            n = np.array((-d[1], d[0])) # normal vector
            n = n/np.linalg.norm(n)
            M = np.eye(2) - np.outer(n, n) # projection matrix

            dlist.append(d)
            nlist.append(n)
            Mlist.append(M)

        self.N = N
        self.nlist = nlist
        self.vertices = vertices
        self.dlist = dlist
        self.Mlist = Mlist

    def project(self, P_in, tol=1e-6, maxit=100):
        """
        Projects a point P onto the polygon

        Parameters
        ----------
        P_in: 2D numpy array
            Point to be projected
        """
        N = self.N
        vertices = self.vertices
        dlist = self.dlist
        Mlist = self.Mlist

        err = 2*tol
        it = 0

        P = P_in.copy()

        while err>tol and it<maxit:
            side_check = np.zeros(N)
            for ii in range(N):
                d = dlist[ii]
                x = vertices[ii]
                side_check[ii] = max(0, d[1]*(x[0] - P[0]) - d[0]*(x[1] - P[1]))
                if side_check[ii]>0:
                    # print('projecting on side ', ii)
                    P = Mlist[ii]@(P - x) + x
            err = np.sum(side_check)
            it += 1
            # print('iteration = ', it)
            # print('point = ', P)
            # print('err = ', err)

        return P

    def plot(self, ax):
        """
        Plots the polygon on provided axes
        """
        vertices = np.array(self.vertices)
        xv = vertices[:, 0]
        yv = vertices[:, 1]
        ax.fill(xv, yv)


def overlapProjector(x1, x2, dmin):
    """
    Projects a pair of points to an admissible distance

    x1: 2D numpy array
        First point
    x2: 2D numpy array
        Second point
    dmin: float
        Minimum distance
    """
    dvec = x2 - x1
    dnorm = np.linalg.norm(dvec)
    dunitvec = dvec/dnorm
    midpoint = 0.5*(x1 + x2)

    if dnorm<dmin:
        x1proj = midpoint - 0.5 * dunitvec * dmin
        x2proj = midpoint + 0.5 * dunitvec * dmin
        return x1proj, x2proj
    else:
        return x1.copy(), x2.copy()


def overlapCheck(x, y, R):
    """
    Checks whether there are overlaps between circles of radius R in
    positions (x, y)
    """
    N = len(x)

    for i in range(N):
        for j in range(i+1, N):
            d2 = ((x[j] - x[i])**2 + (y[j] - y[i])**2)
            if d2 <= 4*R**2:
                print('distance = ', np.sqrt(d2))
                return True

    return False

def overlapIndices(x, y, R):
    """
    Returns the indices of overlapping circles
    """
    N = len(x)
    ind = []

    for i in range(N):
        for j in range(i+1, N):
            d2 = ((x[j] - x[i])**2 + (y[j] - y[i])**2)
            if d2 <= 4*R**2:
                ind.append(i)
                ind.append(j)

    return np.unique(ind)

def overlapPairs(x, y, R):
    """
    Returns the indices of overlapping pairs of circles
    """

    N = len(x)
    ind = []

    for i in range(N):
        for j in range(i+1, N):
            d2 = ((x[j] - x[i])**2 + (y[j] - y[i])**2)
            if d2 <= 4*R**2:
                ind.append((i, j))

    return ind
