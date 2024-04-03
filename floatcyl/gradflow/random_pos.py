#! /usr/bin/env python3

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt


def random_in_square(r, xL, xR, yL, yU, N, maxit=100, seed=None):
    """
    Puts N circles of radius r in a square of edge L, without overlaps
    kwarg: maxit (maximum number of total inner iterations)

    Returns positions x, y and status 0 for not converged, 1 for converged
    """

    max_packing = np.pi * np.sqrt(3) / 6
    if (N * np.pi*r**2/((xR-xL)*(yU-yL))) > max_packing:
        raise ValueError('Impossible packing!')


    rng = default_rng(seed=seed)


    xc = rng.uniform(xL, xR, N)
    yc = rng.uniform(yL, yU, N)

    nit = 0

    reiterate = True

    outer_iter = 0

    while reiterate:

        reiterate = False
        for ii in range(N):
            overlaps = False
            # Check if cylinder ii has overlaps with any other
            for jj in range(N):
                if ii!=jj:
                    dist2 = ((xc[jj] - xc[ii])**2 + (yc[jj] - yc[ii])**2)
                    if dist2 < 4*r**2:
                        overlaps = True
            nit = 0
            while overlaps and nit<maxit:
                nit += 1
                xc[ii] = rng.uniform(xL, xR)
                yc[ii] = rng.uniform(yL, yU)
                overlaps = False
                # Check if cylinder ii has overlaps with any other
                for jj in range(N):
                    if ii!=jj:
                        dist2 = ((xc[jj] - xc[ii])**2 + (yc[jj] - yc[ii])**2)
                        if dist2 < 4*r**2:
                            overlaps = True
            if nit==maxit:
                reiterate = True

        outer_iter += 1


    return xc, yc

def ref_to_physical(x_ref, y_ref, vertices):
    """
    Conversion from reference triangle to physical triangle
    """
    p0 = vertices[0, :]
    l1 = vertices[1, :] - vertices[0, :]
    l2 = vertices[2, :] - vertices[0, :]

    x = p0[0] + x_ref * l1[0] + y_ref * l2[0]
    y = p0[1] + x_ref * l1[1] + y_ref * l2[1]

    return x, y



def random_in_triangle(r, vertices, N, maxit=100, seed=None):
    """
    Puts N circles of radius r in a triangle with specified vertices,
    without overlaps. Uses a reflection algorithm: see
    https://blogs.sas.com/content/iml/2020/10/19/random-points-in-triangle.html
    vertices must be an array of shape (3, 2)
    kwarg: maxit (maximum number of total inner iterations)

    Returns positions x, y
    """

    rng = default_rng(seed=seed)

    x_ref = rng.uniform(0, 1, N)
    y_ref = rng.uniform(0, 1, N)


    for i in range(N):
        if y_ref[i] > (1 - x_ref[i]):
            temp = x_ref[i]
            x_ref[i] = 1 - y_ref[i]
            y_ref[i] = 1 - temp


    x, y = ref_to_physical(x_ref, y_ref, vertices)

    nit = 0
    reiterate = True
    outer_iter = 0

    while reiterate:

        reiterate = False
        for ii in range(N):
            overlaps = False
            # Check if cylinder ii has overlaps with any other
            for jj in range(N):
                if ii!=jj:
                    dist2 = ((x[jj] - x[ii])**2 + (y[jj] - y[ii])**2)
                    if dist2 < 4*r**2:
                        overlaps = True
            nit = 0
            while overlaps and nit<maxit:
                nit += 1
                x_ref[ii] = rng.uniform(0, 1)
                y_ref[ii] = rng.uniform(0, 1)
                if y_ref[ii] > (1 - x_ref[ii]):
                    temp = x_ref[ii]
                    x_ref[ii] = 1 - y_ref[ii]
                    y_ref[ii] = 1 - temp
                x[ii], y[ii] = ref_to_physical(x_ref[ii], y_ref[ii], vertices)
                overlaps = False
                # Check if cylinder ii has overlaps with any other
                for jj in range(N):
                    if ii!=jj:
                        dist2 = ((x[jj] - x[ii])**2 + (y[jj] - y[ii])**2)
                        if dist2 < 4*r**2:
                            overlaps = True
            if nit==maxit:
                reiterate = True

        outer_iter += 1

    return x, y


def random_in_trunc_triangle(r, vertices, N, subd_check, sf=1.2, maxit=10):
    """
    Puts N circles of radius r in the intersection between a triangle
    with specified vertices and a generic domain, without overlaps.

    r: float
        minimum distance between points (radius)
    vertices: array of shape (3, 2)
        triangle vertices
    N: integer
        number of points to be positioned
    subd_check: function
        returns True if a point is inside the subdomain, False otherwise
    sf: float (default = 1.2)
        initial safety factor based on the area ratio between the area of the
        triangle and the subdomain
    maxit: integer (default = 10)
        maximum number of iterations
    """

    N_tot = int(np.ceil(sf*N))
    for ii in range(maxit):
        print('Trying to position ', N_tot, ' points')
        x, y = random_in_triangle(r, vertices, N_tot)
        print('x = ', x)
        print('y = ', y)
        print('Done! Removing unfeasible points')
        ind_remove = []

        for ii in range(N_tot):
            inside = subd_check(np.array((x[ii], y[ii])))
            if not inside:
                ind_remove.append(ii)

        x = np.delete(x, ind_remove)
        y = np.delete(y, ind_remove)

        print('Done! Obtained ', len(x), ' valid points')

        if len(x)>=N:
            print('Removing random points')
            return x[:N], y[:N]
        else:
            N_tot += 1

    raise RuntimeError('Point distribution failed after ', maxit, ' iterations')


def symm_in_trunc_triang(L, R, r, N, arc_origin):
    rmin = 2*(R-r) / np.sqrt(3)
    rmax = np.sqrt(3)/2*L - 3*r
    c = 6 * (N+1) * (rmax-rmin) / (np.pi * (rmax+rmin))
    n = int(np.ceil((-1 + np.sqrt(1 + 4*c)) / 2))
    rvec = np.linspace(rmin, rmax, n)
    dl = 1/(N-1) * np.pi/3 * np.sum(rvec)
    if dl<2*r:
        raise ValueError('Impossible packing! Would require r<'+str(dl/2))
    lvec = np.pi/3 * rvec
    Nvec = np.ceil(np.pi/3 * rvec / dl).astype(int)
    # print('Nvec = ', Nvec)
    # print('total number of points placed = ', int(np.sum(Nvec)))
    dlvec = lvec/Nvec #density of points in each arc
    # print('dlvec = ', dlvec)
    # Remove excess points from last (most populated) arcs
    if int(np.sum(Nvec)) > N:
        Nvec[N-int(np.sum(Nvec)):] -= 1
    xb = np.zeros(N)
    yb = np.zeros(N)
    ind = 0
    for ii in range(n):
        if Nvec[ii] == 1:
            xb[ind] = arc_origin[0] + rvec[ii]
            yb[ind] = arc_origin[1]
            ind +=1
        else:
            xb[ind:ind+Nvec[ii]] = (arc_origin[0] +
                rvec[ii] * np.cos(np.linspace(-np.pi/6, np.pi/6, Nvec[ii])))
            yb[ind:ind+Nvec[ii]] = (arc_origin[1] +
                rvec[ii] * np.sin(np.linspace(-np.pi/6, np.pi/6, Nvec[ii])))
            ind += Nvec[ii]
    return xb, yb
