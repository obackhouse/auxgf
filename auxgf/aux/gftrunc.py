''' Routines for truncation via Green's function moments.
'''

import numpy as np
import scipy.special
import scipy.integrate

from auxgf import util
from auxgf.util import types


#TODO: check memory usage


def kernel(e, nmom, method='power', beta=100):
    ''' Prepares f(e,n), that is the kernel function of the energies
        for a given order required to construct the projector.

    Parameters
    ----------
    e : (m) array
        energies
    nmom : int
        maximum moment order
    method : str, optional
        kernel method {'power', 'legendre'}, default 'power'
    beta : float, optional
        inverse temperature, required for `method='legendre'`,
    
    Returns
    -------
    f_en : (n+1,m) ndarray
        function f(e,n) computed for each `nmom` and `e`
    '''

    e = np.asarray(e).reshape(np.size(e))
    f_en = np.zeros((nmom+1, e.size), dtype=types.float64)

    if method == 'power':
        f_en[0] = 1

        for n in range(1, nmom+1):
            f_en[n] = e ** n

    elif method == 'legendre':
        if beta is None:
            raise ValueError("Must pass keyword argument `beta` for "
                             "auxgf.aux.gftrunc.kernel with `method='power'`")

        tfac = 2.0 / beta

        def fn(t, e, n, beta):
            p = scipy.special.legendre(n)
            x = tfac * t + 1
            return p(x) * np.exp(-e * (t + (e > 0) * beta))

        f_en[0] = 1

        for i in range(e.size):
            for n in range(1, nmom+1):
                f_en[n,i] = scipy.integrate.quad(fn, -beta, 0, args=(e[i], n, beta))[0] 

    return f_en


def build_projector(aux, h_phys, nmom, method='power', beta=100, wtol=1e-10):
    ''' Builds the vectors which project the auxiliary space into a
        compressed one with consistent moments up to order `nmom`.

    Parameters
    ----------
    aux : Aux
        auxiliaries
    h_phys : (n,n) ndarray
        physical space Hamiltonian
    nmom : int
        maximum moment order
    method : str, optional
        kernel method {'power', 'legendre'}, default 'power'
    beta : float, optional
        inverse temperature, required for `method='legendre'`
    wtol : float, optional
        tolerance in eigenvalues for linear dependency analysis

    Returns
    -------
    p : (m,k) ndarray
        projection vectors, the outer-product of which is the projector
    '''

    nphys = aux.nphys

    e, c = aux.eig(h_phys)

    occ = e < aux.chempot
    vir = e >= aux.chempot

    e_occ = kernel(e[occ], nmom, method=method, beta=beta)
    e_vir = kernel(e[vir], nmom, method=method, beta=beta)

    c_occ = c[:,occ]
    c_vir = c[:,vir]

    p_occ = util.einsum('xi,pi,ni->xpn', c_occ[nphys:], c_occ[:nphys], e_occ)
    p_vir = util.einsum('xa,pa,na->xpn', c_vir[nphys:], c_vir[:nphys], e_vir)

    p_occ = p_occ.reshape((aux.naux, aux.nphys*(nmom+1)))
    p_vir = p_vir.reshape((aux.naux, aux.nphys*(nmom+1)))

    p = np.hstack((p_occ, p_vir))

    p = util.normalise(p)
    w, p = util.eigh(np.dot(p, p.T))
    p = p[:, w > wtol]

    p = util.block_diag([np.eye(nphys), p])

    return p


def build_auxiliaries(h, nphys):
    ''' Builds a set of auxiliary energies and couplings from the
        Hamiltonian.

    Parameters
    ----------
    h : (n,n) ndarray
        Hamiltonian
    nphys : int
        number of physical degrees of freedom

    Returns
    -------
    e : (m) ndarray
        auxiliary energies
    v : (nphys,m) ndarray
        auxiliary couplings
    '''

    w, v = util.eigh(h[nphys:,nphys:])

    e = w

    v = np.dot(h[:nphys,nphys:], v)

    return e, v


def run(aux, h_phys, nmom, method='power', beta=100):
    ''' Runs the truncation by moments of the Green's function.

        [1] arXiv:1904.08019

    Parameters
    ----------
    aux : Aux
        auxiliaries
    h_phys : (n,n) ndarray
        physical space Hamiltonian
    nmom : int
        number of moments
    method : str, optional
        kernel method {'power', 'legendre'}, default 'power'
    beta : float, optional
        inverse temperature, required for `method='legendre'`

    Returns
    -------
    red : Aux
        reduced auxiliaries
    '''

    #TODO: debugging mode which checks the moments

    p = build_projector(aux, h_phys, nmom, method=method, beta=beta)

    h_tilde = np.dot(p.T, aux.dot(h_phys, p))

    e, v = build_auxiliaries(h_tilde, aux.nphys)

    red = aux.new(e, v)

    return red
    








