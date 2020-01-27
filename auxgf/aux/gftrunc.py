''' Routines for truncation via Green's function moments.
'''

import numpy as np

from auxgf import util
from auxgf.util import types


#TODO: check memory usage here

def build_projector(aux, h_phys, nmom, wtol=1e-10):
    ''' Builds the vectors which project the auxiliary space into a
        compressed one with consistent moments up to order `nmom`.

    Parameters
    ----------
    aux : Aux
        auxiliaries
    h_phys : (n,n) ndarray
        physical space Hamiltonian
    nmom : int
        number of moments
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

    e_occ = np.power.outer(e[occ], np.arange(nmom+1))
    e_vir = np.power.outer(e[vir], np.arange(nmom+1))

    c_occ = c[:,occ]
    c_vir = c[:,vir]

    p_occ = util.einsum('xi,pi,in->xpn', c_occ[nphys:], c_occ[:nphys], e_occ)
    p_vir = util.einsum('xa,pa,an->xpn', c_vir[nphys:], c_vir[:nphys], e_vir)

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


def run(aux, h_phys, nmom):
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

    Returns
    -------
    red : Aux
        reduced auxiliaries
    '''

    #TODO: debugging mode which checks the moments

    p = build_projector(aux, h_phys, nmom)

    h_tilde = np.dot(p.T, aux.dot(h_phys, p))

    e, v = build_auxiliaries(h_tilde, aux.nphys)

    red = aux.new(e, v)

    return red
    








