''' Routines for building sets of auxiliaries according to the RMP2
    self-energy.
'''

import numpy as np

from auxgf import util, aux
from auxgf.util import types


def _parse_rhf(e, eri, chempot):
    if not np.all(np.diff(e) >= 0):
        # Masking is slower but necessary if energies aren't sorted -
        # they typically should be sorted though and could require it
        # in the future?

        o = e < chempot
        v = e >= chempot

        eo = e[o]
        ev = e[v]

        xija = eri[:,:,:,v][:,:,o][:,o]
        xabi = eri[:,:,:,o][:,:,v][:,v]

    else:
        o = slice(None, np.sum(e < chempot))
        v = slice(np.sum(e < chempot), None)

        eo = e[o]
        ev = e[v]

        xija = eri[:,o,o,v]
        xabi = eri[:,v,v,o]

    return eo, ev, xija, xabi


def make_coups_inner(v, wtol=1e-12):
    ''' Builds a set of couplings using the eigenvectors of the inner
        product of the space spanned by a set of vectors.

    Parameters
    ----------
    v : (n,m) ndarray
        input vectors
    wtol : float, optional
        threshold for an eigenvalue to be considered zero

    Returns
    -------
    coup : (n,k) ndarray
        coupling vectors
    '''

    if v.ndim == 1:
        v = v[:,None]

    s = np.dot(v.T, v)

    #TODO: lib.dsyevd_2x2

    w, coup = util.eigh(s)

    mask = w >= wtol

    coup = np.dot(v, coup[:,mask])
    coup /= np.sqrt(np.sum(coup * coup, axis=0))
    coup *= np.sqrt(w[mask])

    return coup


def make_coups_outer(v, s=None, wtol=1e-12):
    ''' Builds a set of couplings using the eigenvectors of the outer
        product of the space spanned by a set of vectors, scaled by
        their signs.

    Parameters
    ----------
    v : (n,m) ndarray
        input vectors
    s : (m) ndarray, optional
        signs (+1 causal, -1 causal) of vectors
    wtol : float, optional
        threshold for an eigenvalue to be considered zero

    Returns
    -------
    coup : (n,k) ndarray
        coupling vectors
    sign : (k) ndarray, optional
        signs, if `s` is not `None`
    '''

    if v.ndim == 1:
        v = v[:,None]

    if s is None:
        m = np.dot(v, v.T)
    else:
        m = np.dot(s * v, v.T)

    w, coup = util.eigh(m)

    mask = np.absolute(w) >= wtol

    coup = coup[:,mask]
    coup *= np.sqrt(np.absolute(w[mask]))

    if s is None:
        return coup
    else:
        sign = np.sign(w[mask])
        return coup, sign


def build_rmp2_part(eo, ev, xija, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for a restricted reference.

    Parameters
    ----------
    eo : (o) ndarray
        occupied (virtual) energies
    ev : (v) ndarray
        virtual (occupied) energies
    xija : (n,o,o,v)
        two-electron integrals indexed as physical, occupied, occupied,
        virtual (physical, virtual, virtual, occupied)
    wtol : float, optional
        threshold for an eigenvalue to be considered zero
    ss_factor : float, optional
        same spin factor, default 1.0
    os_factor : float, optional
        opposite spin factor, deafult 1.0

    Returns
    -------
    e : (m) ndarray
        auxiliary energies
    v : (n,m) ndarray
        auxiliary couplings
    '''

    nphys, nocc, _, nvir = xija.shape
    npoles = nocc * nocc * nvir

    e = np.zeros((npoles), dtype=types.float64)
    v = np.zeros((nphys, npoles), dtype=types.float64)

    pos_factor = np.sqrt(0.5 * os_factor)
    neg_factor = np.sqrt(0.5 * os_factor + ss_factor)
    dia_factor = np.sqrt(os_factor)

    n0 = 0
    for i in range(nocc):
       nja = i * nvir
       jm = slice(None, i) 
       am = slice(n0, n0+nja)
       bm = slice(n0+nja, n0+nja*2)
       cm = slice(n0+nja*2, n0+nja*2+nvir)

       vija = xija[:,i,jm].reshape((nphys, nja))
       vjia = xija[:,jm,i].reshape((nphys, nja))

       e[am] = eo[i] + np.subtract.outer(eo[jm], ev).flatten()
       e[bm] = e[am]
       e[cm] = 2 * eo[i] - ev

       v[:,am] = neg_factor * (vija - vjia)
       v[:,bm] = pos_factor * (vija + vjia)
       v[:,cm] = dia_factor * xija[:,i,i]

       n0 += nja * 2 + nvir

    mask = np.sum(v*v, axis=0) >= wtol
    e = e[mask]
    v = v[:,mask]

    assert e.shape[0] == v.shape[1]

    return e, v


def build_rmp2(e, eri, chempot=0.0, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for a restricted reference.

    Parameters
    ----------
    e : (n) ndarray
        MO or QMO energies
    eri : (n,m,m,m)
        two-electron integrals where the first index is in the physical
        basis
    chempot : float, optional
        chemical potential
    wtol : float, optional
        threshold for an eigenvalue to be considered zero
    ss_factor : float, optional
        same spin factor, default 1.0
    os_factor : float, optional
        opposite spin factor, deafult 1.0
    
    Returns
    -------
    poles : Aux
        auxiliaries
    '''

    eo, ev, xija, xabi = _parse_rhf(e, eri, chempot)

    eija, vija = build_rmp2_part(eo, ev, xija, wtol=wtol, 
                                 ss_factor=ss_factor, os_factor=os_factor)
    eabi, vabi = build_rmp2_part(ev, eo, xabi, wtol=wtol, 
                                 ss_factor=ss_factor, os_factor=os_factor)

    e = np.concatenate((eija, eabi), axis=0)
    v = np.concatenate((vija, vabi), axis=1)

    poles = aux.Aux(e, v, chempot=chempot)

    return poles


def build_rmp2_iter(aux, h_phys, eri_mo, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams by iterating the current set of auxiliaries according
        to the eigenvalue form of the Dyson equation.

    Parameters
    ----------
    aux : Aux
        auxiliaries of previous iteration
    h_phys : (n,n) ndarray
        physical space hamiltonian
    eri_mo : (n,n,n,n) ndarray
        two-electron repulsion integrals in MO basis
    wtol : float, optional
        threshold for an eigenvalue to be considered zero
    ss_factor : float, optional
        same spin factor, default 1.0
    os_factor : float, optional
        opposite spin factor, deafult 1.0

    Returns
    -------
    poles : Aux
        auxiliaries
    '''

    e, c = aux.eig(h_phys)

    o = e < aux.chempot
    v = e >= aux.chempot

    eo = e[o]
    ev = e[v]
    co = c[:aux.nphys,o]
    cv = c[:aux.nphys,v]

    xija = util.mo2qo(eri_mo, co, co, cv)
    eija, vija = build_rmp2_part(eo, ev, xija, wtol=wtol,
                                 ss_factor=ss_factor, os_factor=os_factor)
    del xija

    xabi = util.mo2qo(eri_mo, cv, cv, co)
    eabi, vabi = build_rmp2_part(ev, eo, xabi, wtol=wtol,
                                 ss_factor=ss_factor, os_factor=os_factor)
    del xabi

    e = np.concatenate((eija, eabi), axis=0)
    v = np.concatenate((vija, vabi), axis=1)

    poles = aux.new(e, v)

    return poles


def build_rmp2_part_direct(eo, ev, xija, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    #TODO: is sharing the memory in these yield statements efficient?

    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for a restricted reference. Uses a generator which
        iterates over blocks.

    Parameters
    ----------
    eo : (o) ndarray
        occupied (virtual) energies
    ev : (v) ndarray
        virtual (occupied) energies
    xija : (n,o,o,v)
        two-electron integrals indexed as physical, occupied, occupied,
        virtual (physical, virtual, virtual, occupied)
    wtol : float, optional
        threshold for an eigenvalue to be considered zero
    ss_factor : float, optional
        same spin factor, default 1.0
    os_factor : float, optional
        opposite spin factor, deafult 1.0

    Yields
    ------
    e : (m) ndarray
        auxiliary energies
    v : (n,m) ndarray
        auxiliary couplings
    '''

    nphys, nocc, _, nvir = xija.shape
    npoles = nocc * nocc * nvir

    e = np.zeros((npoles), dtype=types.float64)
    v = np.zeros((nphys, npoles), dtype=types.float64)

    pos_factor = np.sqrt(0.5 * os_factor)
    neg_factor = np.sqrt(0.5 * os_factor + ss_factor)
    dia_factor = np.sqrt(os_factor)

    n0 = 0
    for i in range(nocc):
       nja = i * nvir
       jm = slice(None, i) 
       am = slice(n0, n0+nja)
       bm = slice(n0+nja, n0+nja*2)
       cm = slice(n0+nja*2, n0+nja*2+nvir)

       vija = xija[:,i,jm].reshape((nphys, nja))
       vjia = xija[:,jm,i].reshape((nphys, nja))

       e[am] = eo[i] + np.subtract.outer(eo[jm], ev).flatten()
       e[bm] = e[am]
       e[cm] = 2 * eo[i] - ev

       v[:,am] = neg_factor * (vija - vjia)
       v[:,bm] = pos_factor * (vija + vjia)
       v[:,cm] = dia_factor * xija[:,i,i]

       n1 = n0 + nja * 2 + nvir

       yield e[n0:n1], v[:,n0:n1]

       n0 = n1


def build_rmp2_direct(e, eri, chempot=0.0, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for a restricted reference. Uses a generator which
        iterates over blocks.

    Parameters
    ----------
    e : (n) ndarray
        MO or QMO energies
    eri : (n,m,m,m)
        two-electron integrals where the first index is in the physical
        basis
    chempot : float, optional
        chemical potential
    wtol : float, optional
        threshold for an eigenvalue to be considered zero
    ss_factor : float, optional
        same spin factor, default 1.0
    os_factor : float, optional
        opposite spin factor, deafult 1.0
    
    Yields
    ------
    poles : Aux
        auxiliaries
    '''
    
    eo, ev, xija, xabi = _parse_rhf(e, eri, chempot)

    kwargs = dict(ss_factor=ss_factor, os_factor=os_factor, wtol=wtol)

    for e,v in build_rmp2_part_direct(eo, ev, xija, **kwargs):
        yield aux.Aux(e, v, chempot=chempot)

    for e,v in build_rmp2_part_direct(ev, eo, xabi, **kwargs):
        yield aux.Aux(e, v, chempot=chempot)


def build_rmp2_part_se_direct(eo, ev, xija, grid, chempot=0.0, ordering='feynman'):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for a restricted reference. Poles are summed straight
        into the self-energy and returns an `ndarray` instead of `Aux`.

    Parameters
    ----------
    eo : (o) ndarray
        occupied (virtual) energies
    ev : (v) ndarray
        virtual (occupied) energies
    xija : (n,o,o,v)
        two-electron integrals indexed as physical, occupied, occupied,
        virtual (physical, virtual, virtual, occupied)
    grid : (k) ImFqGrid, ImFqQuad or ReFqGrid
        grid
    chempot : float, optional
        chemical potential
    ordering : str
        ordering of the poles {'feynman', 'advanced', 'retarded'}
        (default 'feynman')

    Returns
    -------
    se : (k,n,n) ndarray
        frequency-dependent self-energy
    '''

    #TODO write in C

    if grid.axis == 'imag':
        if ordering == 'feynman':
            get_s = lambda x : np.sign(x)
        elif ordering == 'advanced':
            get_s = lambda x : np.ones(x.shape, dtype=types.int64)
        elif ordering == 'retarded':
            get_s = lambda x : -np.ones(x.shape, dtype=types.int64)
    else:
        get_s = lambda x : 0.0

    w = grid.prefac * grid.values

    nphys, nocc, _, nvir = xija.shape
    se = np.zeros((grid.shape[0], nphys, nphys), dtype=types.complex128)

    eov = util.outer_sum([eo, -ev]).flatten() - chempot

    for i in range(nocc):
        ei = eo[i] + eov

        vi = xija[:,i].reshape((nphys, -1))
        vip = xija[:,:,i].reshape((nphys, -1))

        di = 1.0 / util.outer_sum([w, -ei + get_s(ei) * grid.eta * 1.0j])

        se += util.einsum('wk,xk,yk->wxy', di, vi, 2*vi-vip)

    return se


def build_rmp2_se_direct(e, eri, grid, chempot=0.0, ordering='feynman'):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for a restricted reference. Poles are summed straight
        into the self-energy and returns an `ndarray` instead of `Aux`.

    Parameters
    ----------
    e : (n) ndarray
        MO or QMO energies
    eri : (n,m,m,m)
        two-electron integrals where the first index is in the physical
        basis
    grid : (k) ImFqGrid, ImFqQuad or ReFqGrid
        grid
    chempot : float, optional
        chemical potential
    ordering : str
        ordering of the poles {'feynman', 'advanced', 'retarded'}
        (default 'feynman')
    
    Yields
    ------
    se : (k,n,n) ndarray
        frequency-dependent self-energy
    '''

    eo, ev, xija, xabi = _parse_rhf(e, eri, chempot)

    se  = build_rmp2_part_se_direct(eo, ev, xija, grid, chempot=chempot, 
                                    ordering=ordering)
    se += build_rmp2_part_se_direct(ev, eo, xabi, grid, chempot=chempot, 
                                    ordering=ordering)

    return se
