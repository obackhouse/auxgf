''' Routines for truncation via Green's function moments.
'''

import numpy as np
import scipy.special
import scipy.integrate

from auxgf import util
from auxgf.util import types
from auxgf.aux import _gfkern


#TODO: check memory usage


def kernel(e, nmom, method='power', beta=100, chempot=0.0):
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
    chempot : float, optional
        chemical potential, required for `method='legendre'`
    
    Returns
    -------
    f_en : (n+1,m) ndarray
        function f(e,n) computed for each `nmom` and `e`
    '''

    e = np.asarray(e).reshape(np.size(e))
    f_en = np.zeros((nmom+1, e.size), dtype=types.float64)
    f_en[0] = 1

    if method == 'power':
        for n in range(1, nmom+1):
            f_en[n] = e ** n

    elif method == 'legendre':
        if beta is None:
            raise ValueError("Must pass keyword argument `beta` for "
                             "auxgf.aux.gftrunc.kernel with `method='power'`")

        for n in range(1, min(nmom+1, _gfkern._max_kernel_order+1)):
            f_en[n] = _gfkern._legendre_bath_kernel(n, e, beta, chempot=chempot)

        if nmom > _gfkern._max_kernel_order:
            tfac = 2.0 / beta

            def fn(t, e, n, beta):
                p = scipy.special.legendre(n)
                x = tfac * t + 1
                return p(x) * np.exp(-e * (t + (e-chempot > 0) * beta))

            f_en[0] = 1

            for i in range(e.size):
                for n in range(_gfkern._max_kernel_order+1, nmom+1):
                    f_en[n,i] = scipy.integrate.quad(fn, -beta, 0, 
                                                     args=(e[i], n, beta))[0]

    return f_en


def build_projector(se, h_phys, nmom, method='power', beta=100, wtol=1e-12, chempot=0.0):
    ''' Builds the vectors which project the auxiliary space into a
        compressed one with consistent moments up to order `nmom`.

    Parameters
    ----------
    se : Aux
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
    chempot : float, optional
        chemical potential, required for `method='legendre'`

    Returns
    -------
    p : (m,k) ndarray
        projection vectors, the outer-product of which is the projector
    '''

    nphys = se.nphys

    e, c = se.eig(h_phys)

    occ = e < se.chempot
    vir = e >= se.chempot

    e_occ = kernel(e[occ], nmom, method=method, beta=beta, chempot=chempot)
    e_vir = kernel(e[vir], nmom, method=method, beta=beta, chempot=chempot)

    c_occ = c[:,occ]
    c_vir = c[:,vir]

    p_occ = util.einsum('xi,pi,ni->xpn', c_occ[nphys:], c_occ[:nphys], e_occ)
    p_vir = util.einsum('xa,pa,na->xpn', c_vir[nphys:], c_vir[:nphys], e_vir)

    p_occ = p_occ.reshape((se.naux, se.nphys*(nmom+1)))
    p_vir = p_vir.reshape((se.naux, se.nphys*(nmom+1)))

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


def run(se, h_phys, nmom, method='power', beta=100, chempot=0.0):
    ''' Runs the truncation by moments of the Green's function.

        [1] arXiv:1904.08019

    Parameters
    ----------
    se : Aux
        auxiliaries
    h_phys : (n,n) ndarray
        physical space Hamiltonian
    nmom : int
        number of moments
    method : str, optional
        kernel method {'power', 'legendre'}, default 'power'
    beta : float, optional
        inverse temperature, required for `method='legendre'`
    chempot : float, optional
        chemical potential, required for `method='legendre'`

    Returns
    -------
    red : Aux
        reduced auxiliaries
    '''

    #TODO: debugging mode which checks the moments

    if method == 'legendre': #FIXME?
        raise NotImplementedError

    p = build_projector(se, h_phys, nmom, method=method, 
                        beta=beta, chempot=chempot)

    h_tilde = np.dot(p.T, se.dot(h_phys, p))

    e, v = build_auxiliaries(h_tilde, se.nphys)

    red = se.new(e, v)

    return red
