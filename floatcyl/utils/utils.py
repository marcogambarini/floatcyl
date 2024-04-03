#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaincc, gammainccinv

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

def discrete_PM_spectrum(Te, Hs, Nbins, filename=None, thres=1e-3,
                            plot=False, crit='E'):
    """
    Builds a discretization of a Pierson-Moskowitz spectrum with
    specified parameters

    Parameters
    ----------
    Te: float
        Energy period (s)
    Hs: float
        Significant wave height (m)
    Nbins: integer
        Number of bins (discrete frequencies)
    filename: string or None
        If None, no output file generated; otherwise output file name
        (default: None)
    thres: float
        Threshold for truncation of the spectrum in terms neglected power
        (default: 0.001)
    plot: boolean
        Whether to plot the spectrum and its discretization (default: False)
    crit: string
        Whether to use frequency bins of equal width (F), equal
        energy density (E) (default), or equal energy flux (J)

    Returns
    -------
    omega: array
        Angular frequencies of components of discretized spectrum (rad/s)
    amplitude: array
        Amplitudes of components of discretized spectrum (m)
    """

    # conversion to peak period Tp for P-M spectrum (Guillou2020)
    alpha = 0.86
    Tp = Te/alpha

    fp = 1/Tp
    b = 5/4 * fp**4
    a = b * Hs**2 / 4

    # total power
    P_tot = Hs**2 / 16

    S_PM = lambda f: a/f**5 * np.exp(-b/f**4)
    Pcumfun = lambda f: a/(4*b) * np.exp(-b/f**4)

    # find frequencies realizing the required tolerance on neglected power
    f_L = (-b/np.log(thres/2))**0.25
    f_R = (-b/np.log(1-thres/2))**0.25

    P_negl = P_tot - (Pcumfun(f_R) - Pcumfun(f_L))

    # Discretize the spectrum
    bin_edges = np.zeros(Nbins+1)
    bin_edges[0] = f_L
    bin_edges[-1] = f_R
    bin_centers = np.zeros(Nbins)
    bin_H = np.zeros(Nbins)
    if crit=='F':   # all bins have same width
        bin_edges = np.linspace(f_L, f_R, Nbins+1)
        last_Pcum = Pcumfun(bin_edges[0])
        for ii in range(Nbins):
            P_cum = Pcumfun(bin_edges[ii+1])
            bin_H[ii] = 2 * np.sqrt(2 * (P_cum - last_Pcum))
            last_Pcum = P_cum
    else:
        invPcum = lambda x: (-1/b * np.log(4*b*x/a))**(-0.25)


        if crit=='E': # all bins have same power density
            p_bin = (P_tot - P_negl)/Nbins #power per bin
            bin_H = 2 * np.sqrt(2 * P_tot/Nbins) * np.ones(Nbins)
            last_Pcum = Pcumfun(bin_edges[0])
            for ii in range(Nbins):
                bin_edges[ii+1] = invPcum(p_bin + last_Pcum)
                last_Pcum = Pcumfun(bin_edges[ii+1])
        elif crit=='J': # all bins have same power flux
            # total energy flux
            J_tot = Hs**2 * Te / 16
            # Gamma function constant evaluation
            Jconst = a/(4*b**1.25) * gamma(1.25)

            Jcumfun = lambda x: Jconst * gammaincc(1.25, b/x**4)
            j_bin = (Jcumfun(f_R) - Jcumfun(f_L))/Nbins
            last_Jcum = Jcumfun(bin_edges[0])
            # In this case, all bins have different powers
            for ii in range(Nbins):
                bin_edges[ii+1] = (b/gammainccinv(1.25,
                                    (j_bin + last_Jcum)/Jconst))**0.25
                Jcum = Jcumfun(bin_edges[ii+1])
                fi = 0.5*(bin_edges[ii] + bin_edges[ii+1])
                bin_H[ii] = 2 * np.sqrt(2 * fi * j_bin)
                last_Jcum = Jcum
        else:
            raise ValueError('Criteria for spectrum disc can be F, E or J')

    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    omega = bin_centers*2*np.pi
    amplitude = bin_H/2

    if filename is not None:
        outzfile = filename + '.npz'
        np.savez(outzfile,
                omega=omega,
                amplitude=amplitude)

    if plot:
        def P_disc_cum_rel(f, bin_H):
            pc = 0
            for ii in range(Nbins):
                pc = pc + bin_H[ii]**2 / 8 * (f>bin_edges[ii])
            return pc/P_tot

        x = np.linspace(1e-4, 1.3*f_R, 500)


        # Power distribution plot (pdf)
        fig, ax = plt.subplots()
        ax.plot(x, S_PM(x), 'k', linewidth=1.5)
        ax.axvline(x=f_L, linestyle='--', color='k')
        ax.axvline(x=f_R, linestyle='--', color='k')

        ax.set_xlabel('f (Hz)')
        ax.set_ylabel('S(f) (m$^2$ s)')


        # Power cumulative distribution plot (cdf)
        fig, ax = plt.subplots()
        ax.plot(x, Pcumfun(x)/P_tot, label='Fit',
                    linestyle='-')
        ax.plot(x, P_disc_cum_rel(x, bin_H), label='Discretization',
                    linestyle='--')
        ax.plot(bin_centers, P_disc_cum_rel(bin_centers, bin_H), 'o',
                    label='Bins')

        ax.set_xlabel('f (Hz)')
        ax.set_ylabel('C(f) (m$^2$)')
        ax.legend()

        plt.show()


    return omega, amplitude
