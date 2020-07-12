''' Routines for building sets of auxiliaries according to the RMP2
    self-energy for density-fitted two-electron integrals.
'''

import numpy as np

from auxgf import util, aux
from auxgf.util import types


_reshape_internal = lambda x, s1, swap, s2 : \
                           x.reshape(s1).swapaxes(*swap).reshape(s2)

def _parse_rhf(e, qpx, qyz, chempot):
    if not np.all(np.diff(e) >= 0):
        o = e < chempot
        v = e >= chempot

        eo = e[o]
        ev = e[v]

        ixq = qpx[:,:,o].transpose((2,1,0))
        qja = qyz[:,o][:,:,v]

        axq = qpx[:,:,v].transpose((2,1,0))
        qbi = qyz[:,v][:,:,o]

    else:
        o = slice(None, np.sum(e < chempot))
        v = slice(np.sum(e < chempot), None)

        eo = e[o]
        ev = e[v]

        ixq = qpx[:,:,o].transpose((2,1,0))
        qja = qyz[:,o,v]

        axq = qpx[:,:,v].transpose((2,1,0))
        qbi = qyz[:,v,o]

    return eo, ev, ixq, qja, axq, qbi


def build_dfrmp2_part(eo, ev, ixq, qja, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for a restricted reference.

    Parameters
    ----------
    eo : (o) ndarray
        occupied (virtual) energies
    ev : (v) ndarray
        virtual (occupied) energies
    ixq : (o,n,q) ndarray
        density-fitted two-electron integrals indexed as occupied, 
        physical, auxiliary (physical, virtual, auxiliary)
    qja : (q,o,v) ndarray
        density-fitted two-electron integrals indexed as auxiliary,
        occupied, virtual (auxiliary, virtual, occupied)
    wtol : float, optional
        threshold for an eigenvalue to be considered zero
    ss_factor : float, optional
        same spin factor, default 1.0
    os_factor : float, optional
        opposite spin factor, default 1.0

    Returns
    -------
    e : (m) ndarray
        auxiliary energies
    v : (n,m) ndarray
        auxiliary couplings
    '''

    nphys = ixq.shape[1]
    ndf, nocc, nvir = qja.shape
    npoles = nocc * nocc * nvir

    e = np.zeros((npoles), dtype=types.float64)
    v = np.zeros((nphys, npoles), dtype=types.float64)

    pos_factor = np.sqrt(0.5 * os_factor)
    neg_factor = np.sqrt(0.5 * os_factor + ss_factor)
    dia_factor = np.sqrt(os_factor)

    ixq = ixq.reshape((nocc*nphys, ndf))
    qja = qja.reshape((ndf, nocc*nvir))

    n0 = 0
    for i in range(nocc):
        nja = i * nvir
        am = slice(n0, n0+nja)
        bm = slice(n0+nja, n0+nja*2)
        cm = slice(n0+nja*2, n0+nja*2+nvir)

        xq = ixq[i*nphys:(i+1)*nphys]
        qa = qja[:,i*nvir:(i+1)*nvir]

        xja = np.dot(ixq[:i*nphys], qa)
        xja = _reshape_internal(xja, (i, nphys, nvir), (0,1), (nphys, i*nvir))
        xia = np.dot(xq, qja[:,:i*nvir]).reshape((nphys, -1))
        xa = np.dot(xq, qa)

        e[am] = e[bm] = eo[i] + util.dirsum('i,a->ia', eo[:i], -ev).ravel()
        e[cm] = 2 * eo[i] - ev

        v[:,am] = neg_factor * (xja - xia)
        v[:,bm] = pos_factor * (xja + xia)
        v[:,cm] = dia_factor * xa

        n0 += nja * 2 + nvir

    mask = np.sum(v*v, axis=0) >= wtol
    e = e[mask]
    v = v[:,mask]

    assert e.shape[0] == v.shape[1]

    return e, v


def build_dfrmp2(e, qpx, qyz, chempot=0.0, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for a restricted reference.

    Parameters
    ----------
    e : (n) ndarray
        MO or QMO energies
    qpx : (q,p,x) ndarray
        density-fitted two-electron integrals where first index is in
        the physical basis
    qyz : (q,y,z) ndarray
        density-fitted two-electron integrals
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

    eo, ev, ixq, qja, axq, qbi = _parse_rhf(e, qpx, qyz, chempot)

    eija, vija = build_dfrmp2_part(eo, ev, ixq, qja, wtol=wtol,
                                   ss_factor=ss_factor, os_factor=os_factor)
    eabi, vabi = build_dfrmp2_part(ev, eo, axq, qbi, wtol=wtol,
                                   ss_factor=ss_factor, os_factor=os_factor)

    e = np.concatenate((eija, eabi), axis=0)
    v = np.concatenate((vija, vabi), axis=1)

    poles = aux.Aux(e, v, chempot=chempot)

    return poles


def build_dfrmp2_iter(aux, h_phys, eri_mo, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams by iterating the current set of auxiliaries according
        to the eigenvalue form of the Dyson equation.

    Parameters
    ----------
    aux : Aux
        auxiliaries of previous iteration
    h_phys : (n,n) ndarray
        physical space hamiltonian
    eri_mo : (q,n,n) ndarray
        density-fitted two-electron repulsion integrals in MO basis
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

    nphys = aux.nphys
    nocc = np.sum(o)
    nvir = np.sum(v)

    eo = e[o]
    ev = e[v]
    co = c[:aux.nphys,o]
    cv = c[:aux.nphys,v]
    eye = np.eye(aux.nphys)

    ixq = util.ao2mo_df(eri_mo, co, eye)
    ixq = _reshape_internal(ixq, (-1, nocc*nphys), (0,1), (nocc, nphys, -1))
    qja = util.ao2mo_df(eri_mo, co, cv)
    eija, vija = build_dfrmp2_part(eo, ev, ixq, qja, wtol=wtol,
                                   ss_factor=ss_factor, os_factor=os_factor)
    del ixq, qja

    axq = util.ao2mo_df(eri_mo, cv, eye)
    axq = _reshape_internal(axq, (-1, nvir*nphys), (0,1), (nvir, nphys, -1))
    qbi = util.ao2mo_df(eri_mo, cv, co)
    eabi, vabi = build_dfrmp2_part(ev, eo, axq, qbi, wtol=wtol,
                                   ss_factor=ss_factor, os_factor=os_factor)
    del axq, qbi

    e = np.concatenate((eija, eabi), axis=0)
    v = np.concatenate((vija, vabi), axis=1)

    poles = aux.new(e, v)

    return poles

    
def build_dfrmp2_part_direct(eo, ev, ixq, qja, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for a restricted reference. Uses a generator which
        iterates over blocks.

    Parameters
    ----------
    eo : (o) ndarray
        occupied (virtual) energies
    ev : (v) ndarray
        virtual (occupied) energies
    ixq : (o,n,q) ndarray
        density-fitted two-electron integrals indexed as occupied, 
        physical, auxiliary (physical, virtual, auxiliary)
    qja : (q,o,v) ndarray
        density-fitted two-electron integrals indexed as auxiliary,
        occupied, virtual (auxiliary, virtual, occupied)
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

    nphys = ixq.shape[1]
    ndf, nocc, nvir = qja.shape
    npoles = nocc * nocc * nvir

    pos_factor = np.sqrt(0.5 * os_factor)
    neg_factor = np.sqrt(0.5 * os_factor + ss_factor)
    dia_factor = np.sqrt(os_factor)

    ixq = ixq.reshape((nocc*nphys, ndf))
    qja = qja.reshape((ndf, nocc*nvir))

    for i in range(nocc):
        nja = i * nvir

        xq = ixq[i*nphys:(i+1)*nphys]
        qa = qja[:,i*nvir:(i+1)*nvir]

        xja = np.dot(ixq[:i*nphys], qa)
        xja = _reshape_internal(xja, (i, nphys, nvir), (0,1), (nphys, i*nvir))
        xia = np.dot(xq, qja[:,:i*nvir]).reshape((nphys, -1))
        xa = np.dot(xq, qa)

        ea = eb = eo[i] + util.dirsum('i,a->ia', eo[:i], -ev).ravel()
        ec = 2 * eo[i] - ev

        va = neg_factor * (xja - xia)
        vb = pos_factor * (xja + xia)
        vc = dia_factor * xa

        if len(ea):
            yield ea, va
        if len(eb):
            yield eb, vb
        if len(ec):
            yield ec, vc


def build_dfrmp2_direct(e, qpx, qyz, chempot=0.0, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for a restricted reference. Uses a generator which
        iterates over blocks.

    Parameters
    ----------
    e : (n) ndarray
        MO or QMO energies
    qpx : (q,p,x) ndarray
        density-fitted two-electron integrals where first index is in
        the physical basis
    qyz : (q,y,z) ndarray
        density-fitted two-electron integrals
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

    eo, ev, ixq, qja, axq, qbi = _parse_rhf(e, qpx, qyz, chempot)

    kwargs = dict(ss_factor=ss_factor, os_factor=os_factor, wtol=wtol)

    for e,v in build_dfrmp2_part_direct(eo, ev, ixq, qja, **kwargs):
        yield aux.Aux(e, v, chempot=chempot)

    for e,v in build_dfrmp2_part_direct(ev, eo, axq, qbi, **kwargs):
        yield aux.Aux(e, v, chempot=chempot)


def build_dfrmp2_part_se_direct(eo, ev, ixq, qja, grid, chempot=0.0, ordering='feynman'):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for a restricted reference. Poles are summed straight
        into the self-energy and returns an `ndarray` instead of `Aux`.

    Parameters
    ----------
    eo : (o) ndarray
        occupied (virtual) energies
    ev : (v) ndarray
        virtual (occupied) energies
    ixq : (o,n,q) ndarray
        density-fitted two-electron integrals indexed as occupied, 
        physical, auxiliary (physical, virtual, auxiliary)
    qja : (q,o,v) ndarray
        density-fitted two-electron integrals indexed as auxiliary,
        occupied, virtual (auxiliary, virtual, occupied)
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

    nphys = ixq.shape[1]
    ndf, nocc, nvir = qja.shape
    npoles = nocc * nocc * nvir

    se = np.zeros((grid.shape[0], nphys, nphys), dtype=types.complex128)

    ixq = ixq.reshape((nocc*nphys, ndf))
    qja = qja.reshape((ndf, nocc*nvir))

    eov = util.outer_sum([eo, -ev]).flatten()

    for i in range(nocc):
        ei = eo[i] + eov - chempot

        vi = np.dot(ixq[i*nphys:(i+1)*nphys], qja).reshape((nphys, -1))
        vip = np.dot(ixq, qja[:,i*nvir:(i+1)*nvir])
        vip = _reshape_internal(vip, (nocc, nphys, -1), (0,1), (nphys, -1))

        di = 1.0 / util.outer_sum([w, -ei + get_s(ei) * grid.eta * 1.0j])

        se += util.einsum('wk,xk,yk->wxy', di, vi, 2*vi-vip)

    return se


def build_dfrmp2_se_direct(e, qpx, qyz, grid, chempot=0.0, ordering='feynman'):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for a restricted reference. Poles are summed straight
        into the self-energy and returns an `ndarray` instead of `Aux`.

    Parameters
    ----------
    e : (n) ndarray
        MO or QMO energies
    qpx : (q,p,x) ndarray
        density-fitted two-electron integrals where first index is in
        the physical basis
    qyz : (q,y,z) ndarray
        density-fitted two-electron integrals
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

    eo, ev, ixq, qja, axq, qbi = _parse_rhf(e, qpx, qyz, chempot)

    se  = build_dfrmp2_part_se_direct(eo, ev, ixq, qja, grid, chempot=chempot,
                                      ordering=ordering)
    se += build_dfrmp2_part_se_direct(ev, eo, axq, qbi, grid, chempot=chempot, 
                                      ordering=ordering)

    return se
