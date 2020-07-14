''' Routines for building sets of auxiliaries according to the UMP2
    self-energy for density-fitted two-electron integrals.
'''

import numpy as np

from auxgf import util, aux
from auxgf.util import types


_is_tuple = lambda x : isinstance(x, (tuple, list, np.ndarray))

_reshape_internal = lambda x, s1, swap, s2 : \
                           x.reshape(s1).swapaxes(*swap).reshape(s2)


def _parse_uhf(e, qpx, qyz, chempot):
    if not (np.all(np.diff(e[0]) >= 0) and np.all(np.diff(e[1])) >= 0):
        # See auxgf.aux.rmp2._parse_rhf

        oa = e[0] < chempot[0]
        va = e[0] >= chempot[0]
        ob = e[1] < chempot[1]
        vb = e[1] >= chempot[1]

        eo = (e[0][oa], e[1][ob])
        ev = (e[0][va], e[1][vb])

        ixq_a = qpx[0][:,:,oa].transpose((2,1,0))
        qja_a = qyz[0][:,oa][:,:,va]
        qja_b = qyz[1][:,ob][:,:,vb]

        axq_a = qpx[0][:,:,va].transpose((2,1,0))
        qbi_a = qyz[0][:,va][:,:,oa]
        qbi_b = qyz[1][:,vb][:,:,ob]

    else:
        oa = slice(None, np.sum(e[0] < chempot[0]))
        va = slice(np.sum(e[0] < chempot[0]), None)
        ob = slice(None, np.sum(e[1] < chempot[1]))
        vb = slice(np.sum(e[1] < chempot[1]), None)

        eo = (e[0][oa], e[1][ob])
        ev = (e[0][va], e[1][vb])

        ixq_a = qpx[0][:,:,oa].transpose((2,1,0))
        qja_a = qyz[0][:,oa,va]
        qja_b = qyz[1][:,ob,vb]

        axq_a = qpx[0][:,:,va].transpose((2,1,0))
        qbi_a = qyz[0][:,va,oa]
        qbi_b = qyz[1][:,vb,ob]

    return eo, ev, (ixq_a, None), (qja_a, qja_b), (axq_a, None), (qbi_a, qbi_b)


def build_dfump2_part(eo, ev, ixq, qja, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for an unrestricted reference.

    Parameters
    ----------
    eo : 2-tuple of (o) ndarray
        occupied (virtual) energies for alpha, beta (beta, alpha) spin
    ev : 2-tuple of (v) ndarray
        virtual (occupied) energies for alpha, beta (beta, alpha) spin
    ixq : 1-tuple or 2-tuple of (n,o,o,v)
        density-fitted two-electron integrals index as occupied,
        physical, auxiliary (virtual, physical, auxiliary) for alpha,
        beta (beta, alpha) spin. Only alpha (beta) is required.
    qja : 2-tuple of (n,o,o,v)
        density-fitted two-electron integrals index as auxiliary,
        occupied, virtual (auxiliary, virtual, occupied) for alpha,
        beta (beta, alpha) spin.
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

    nphys = ixq[0].shape[1]
    ndf, nocca, nvira = qja[0].shape
    _, noccb, nvirb = qja[1].shape
    npoles  = nvira * nocca * (nocca-1) // 2
    npoles += nvirb * nocca * noccb

    e = np.zeros((npoles), dtype=types.float64)
    v = np.zeros((nphys, npoles), dtype=types.float64)

    a_factor = np.sqrt(ss_factor)
    b_factor = np.sqrt(os_factor)

    ixq = (ixq[0].reshape((nocca*nphys, ndf)),)
    qja = (qja[0].reshape((ndf, nocca*nvira)), 
           qja[1].reshape((ndf, noccb*nvirb)))

    n0 = 0
    for i in range(nocca):
        nja_a = i * nvira
        nja_b = noccb * nvirb
        jm = slice(None, i)
        am = slice(n0, n0+nja_a)
        bm = slice(n0+nja_a, n0+nja_a+nja_b)

        xq_a = ixq[0][i*nphys:(i+1)*nphys]
        qa_a = qja[0][:,i*nvira:(i+1)*nvira]

        xja_aa = np.dot(ixq[0][:i*nphys], qa_a)
        xja_aa = _reshape_internal(xja_aa, (i, nphys, nvira), (0,1), (nphys, i*nvira))
        xia_aa = np.dot(xq_a, qja[0][:,:i*nvira]).reshape((nphys,-1))
        xja_ab = np.dot(xq_a, qja[1]).reshape((nphys,-1))

        xija_aa = np.dot(ixq[0], qja[0]).reshape((nocca, nphys, nocca, nvira)).swapaxes(0,1)
        xija_ab = np.dot(ixq[0], qja[1]).reshape((nocca, nphys, noccb, nvirb)).swapaxes(0,1)
        assert np.allclose(xia_aa, xija_aa[:,i,:i].reshape((nphys, -1)))
        assert np.allclose(xja_aa, xija_aa[:,:i,i].reshape((nphys, -1)))
        assert np.allclose(xja_ab, xija_ab[:,i].reshape((nphys, -1)))

        e[am] = eo[0][i] + np.subtract.outer(eo[0][jm], ev[0]).flatten()
        e[bm] = eo[0][i] + np.subtract.outer(eo[1], ev[1]).flatten()

        v[:,am] = a_factor * (xia_aa - xja_aa)
        v[:,bm] = b_factor * xja_ab

        n0 += nja_a + nja_b

    mask = np.sum(v*v, axis=0) >= wtol
    e = e[mask]
    v = v[:,mask]

    e = e[:n0]
    v = v[:,:n0]

    assert e.shape[0] == v.shape[1]

    return e, v


def build_dfump2(e, qpx, qyz, chempot=0.0, **kwargs):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for an unrestricted reference.

    Parameters
    ----------
    e : tuple of (n) ndarray
        MO or QMO energies for alpha, beta (beta, alpha) spin
    qpx : 2-tuple of (q,p,x) ndarray
        density-fitted two-electron integrals where first index is in
        the physical basis for alpha, beta (beta, alpha) spin
    qyz : 2-tuple of (q,y,z) ndarray
        density-fitted two-electron integrals for alpha, beta (beta,
        alpha) spin
    chempot : tuple of float, optional
        chemical potential for alpha, beta (beta, alpha) spin
    wtol : float, optional
        threshold for an eigenvalue to be considered zero
    ss_factor : float, optional
        same spin factor, default 1.0
    os_factor : float, optional
        opposite spin factor, default 1.0
    
    Returns
    -------
    poles : Aux
        auxiliaries
    '''

    if not _is_tuple(chempot):
        chempot = (chempot, chempot)

    eo, ev, ixq, qja, axq, qbi = _parse_uhf(e, qpx, qyz, chempot)

    eija, vija = build_dfump2_part(eo, ev, ixq, qja, **kwargs) 
    eabi, vabi = build_dfump2_part(ev, eo, axq, qbi, **kwargs) 

    e = np.concatenate((eija, eabi), axis=0)
    v = np.concatenate((vija, vabi), axis=1)

    poles = aux.Aux(e, v, chempot=chempot[0])

    return poles


def build_dfump2_iter(se, h_phys, eri_mo, **kwargs):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams by iterating the current set of auxiliaries according
        to the eigenvalue form of the Dyson equation.

        Unlike the other routines in this module, this builds for both
        spins.

    Parameters
    ----------
    aux : tuple of Aux
        auxiliaries of previous iteration as (alpha, beta)
    h_phys : (2,n,n) ndarray
        physical space Hamiltonian for alpha and beta spin, if `ndim==2`
        then spin-symmetry in the Hamiltonian is assumed
    eri_mo : (2,q,n,n) ndarray
        density-fitted two-electron repulsion integrals in MO basis, 
        where the first index is spin
    wtol : float, optional
        threshold for an eigenvalue to be considered zero
    ss_factor : float, optional
        same spin factor, default 1.0
    os_factor : float, optional
        opposite spin factor, default 1.0

    Returns
    -------
    poles_a : Aux
        auxiliaries for alpha spin
    poles_b : Aux
        auxiliaries for beta spin
    '''

    h_phys = np.asarray(h_phys, dtype=types.float64)

    if h_phys.ndim == 2:
        h_phys = np.stack((h_phys, h_phys))

    ea, ca = se[0].eig(h_phys[0])
    eb, cb = se[1].eig(h_phys[1])

    oa = ea < se[0].chempot
    ob = eb < se[1].chempot
    va = ea >= se[0].chempot
    vb = eb >= se[1].chempot

    nphys = se[0].nphys
    nocca = np.sum(oa)
    noccb = np.sum(ob)
    nvira = np.sum(va)
    nvirb = np.sum(vb)

    eo = (ea[oa], eb[ob])
    ev = (ea[va], eb[vb])
    co = (ca[:se[0].nphys,oa], cb[:se[1].nphys,ob])
    cv = (ca[:se[0].nphys,va], cb[:se[1].nphys,vb])
    eye = np.eye(nphys)

    ixq_a = util.ao2mo_df(eri_mo[0], co[0], eye)
    ixq_a = _reshape_internal(ixq_a, (-1, nocca*nphys), (0,1), (nocca, nphys, -1))
    qja_a = util.ao2mo_df(eri_mo[0], co[0], cv[0])
    qja_b = util.ao2mo_df(eri_mo[1], co[1], cv[1])
    eija_a, vija_a = build_dfump2_part(eo, ev, (ixq_a,), (qja_a, qja_b), **kwargs)
    del ixq_a

    ixq_b = util.ao2mo_df(eri_mo[1], co[1], eye)
    ixq_b = _reshape_internal(ixq_b, (-1, noccb*nphys), (0,1), (noccb, nphys, -1))
    eija_b, vija_b = build_dfump2_part(eo[::-1], ev[::-1], (ixq_b,), (qja_b, qja_a), **kwargs)
    del ixq_b, qja_a, qja_b

    axq_a = util.ao2mo_df(eri_mo[0], cv[0], eye)
    axq_a = _reshape_internal(axq_a, (-1, nvira*nphys), (0,1), (nvira, nphys, -1))
    qbi_a = util.ao2mo_df(eri_mo[0], cv[0], co[0])
    qbi_b = util.ao2mo_df(eri_mo[1], cv[1], co[1])
    eabi_a, vabi_a = build_dfump2_part(ev, eo, (axq_a,), (qbi_a, qbi_b), **kwargs)
    del axq_a

    axq_b = util.ao2mo_df(eri_mo[1], cv[1], eye)
    axq_b = _reshape_internal(axq_b, (-1, nvirb*nphys), (0,1), (nvirb, nphys, -1))
    eabi_b, vabi_b = build_dfump2_part(ev[::-1], eo[::-1], (axq_b,), (qbi_b, qbi_a), **kwargs)
    del axq_b, qbi_a, qbi_b

    ea = np.concatenate((eija_a, eabi_a), axis=0)
    eb = np.concatenate((eija_b, eabi_b), axis=0)
    va = np.concatenate((vija_a, vabi_a), axis=1)
    vb = np.concatenate((vija_b, vabi_b), axis=1)

    poles_a = se[0].new(ea, va)
    poles_b = se[1].new(eb, vb)

    return poles_a, poles_b


def build_dfump2_part_direct(eo, ev, ixq, qja, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for an unrestricted reference. Uses a generator which
        iterates over blocks.

    Parameters
    ----------
    eo : 2-tuple of (o) ndarray
        occupied (virtual) energies for alpha, beta (beta, alpha) spin
    ev : 2-tuple of (v) ndarray
        virtual (occupied) energies for alpha, beta (beta, alpha) spin
    ixq : 1-tuple or 2-tuple of (n,o,o,v)
        density-fitted two-electron integrals index as occupied,
        physical, auxiliary (virtual, physical, auxiliary) for alpha,
        beta (beta, alpha) spin. Only alpha (beta) is required.
    qja : 2-tuple of (n,o,o,v)
        density-fitted two-electron integrals index as auxiliary,
        occupied, virtual (auxiliary, virtual, occupied) for alpha,
        beta (beta, alpha) spin.
    wtol : float, optional
        threshold for an eigenvalue to be considered zero
    ss_factor : float, optional
        same spin factor, default 1.0
    os_factor : float, optional
        opposite spin factor, default 1.0

    Yields
    ------
    e : (m) ndarray
        auxiliary energies
    v : (n,m) ndarray
        auxiliary couplings
    '''

    nphys = ixq[0].shape[1]
    ndf, nocca, nvira = qja[0].shape
    _, noccb, nvirb = qja[1].shape
    npoles  = nvira * nocca * (nocca-1) // 2
    npoles += nvirb * nocca * noccb

    a_factor = np.sqrt(ss_factor)
    b_factor = np.sqrt(os_factor)

    ixq = (ixq[0].reshape((nocca*nphys, ndf)),)
    qja = (qja[0].reshape((ndf, nocca*nvira)), qja[1].reshape((ndf, noccb*nvirb)))

    for i in range(nocca):
        nja_a = i * nvira
        nja_b = noccb * nvirb
        jm = slice(None, i)

        xq_a = ixq[0][i*nphys:(i+1)*nphys]
        qa_a = qja[0][:,i*nvira:(i+1)*nvira]

        xja_aa = np.dot(ixq[0][:i*nphys], qa_a)
        xja_aa = _reshape_internal(xja_aa, (i, nphys, nvira), (0,1), (nphys, i*nvira))
        xia_aa = np.dot(xq_a, qja[0][:,:i*nvira]).reshape((nphys,-1))
        xja_ab = np.dot(ixq[0][i*nphys:(i+1)*nphys], qja[1]).reshape((nphys,-1))

        ea = eo[0][i] + np.subtract.outer(eo[0][jm], ev[0]).flatten()
        eb = eo[0][i] + np.subtract.outer(eo[1], ev[1]).flatten()

        va = a_factor * (xja_aa - xia_aa)
        vb = b_factor * xja_ab

        if len(ea):
            yield ea, va
        if len(eb):
            yield eb, vb


def build_dfump2_direct(e, qpx, qyz, chempot=0.0, **kwargs):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for an unrestricted reference. Uses a generator which
        iterates over blocks.

    Parameters
    ----------
    e : tuple of (n) ndarray
        MO or QMO energies for alpha, beta (beta, alpha) spin
    qpx : 2-tuple of (q,p,x) ndarray
        density-fitted two-electron integrals where first index is in
        the physical basis for alpha, beta (beta, alpha) spin
    qyz : 2-tuple of (q,y,z) ndarray
        density-fitted two-electron integrals for alpha, beta (beta,
        alpha) spin
    chempot : tuple of float, optional
        chemical potential for alpha, beta (beta, alpha) spin
    wtol : float, optional
        threshold for an eigenvalue to be considered zero
    ss_factor : float, optional
        same spin factor, default 1.0
    os_factor : float, optional
        opposite spin factor, default 1.0
    
    Yields
    ------
    poles_a : Aux
        auxiliaries for alpha spin
    poles_b : Aux
        auxiliaries for beta spin
    '''

    if not _is_tuple(chempot):
        chempot = (chempot, chempot)
    
    eo, ev, ixq, qja, axq, qbi = _parse_uhf(e, qpx, qyz, chempot)

    for e,v in build_dfump2_part_direct(eo, ev, ixq, qja, **kwargs): 
        yield aux.Aux(e, v, chempot=chempot[0])

    for e,v in build_dfump2_part_direct(ev, eo, axq, qbi, **kwargs): 
        yield aux.Aux(e, v, chempot=chempot[1])


def build_dfump2_part_se_direct(eo, ev, ixq, qja, grid, chempot=0.0, ordering='feynman'):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for an unrestricted reference. Poles are summed straight
        into the self-energy and returns an `ndarray` instead of `Aux`.

    Parameters
    ----------
    eo : (o) ndarray
        occupied (virtual) energies
    ev : (v) ndarray
        virtual (occupied) energies
    ixq : 1-tuple or 2-tuple of (n,o,o,v)
        density-fitted two-electron integrals index as occupied,
        physical, auxiliary (virtual, physical, auxiliary) for alpha,
        beta (beta, alpha) spin. Only alpha (beta) is required.
    qja : 2-tuple of (n,o,o,v)
        density-fitted two-electron integrals index as auxiliary,
        occupied, virtual (auxiliary, virtual, occupied) for alpha,
        beta (beta, alpha) spin.
    grid : (k) ImFqGrid, ImFqQuad or ReFqGrid
        grid
    chempot : tuple of float, optional
        chemical potential for alpha, beta (beta, alpha) spin
    ordering : str
        ordering of the poles {'feynman', 'advanced', 'retarded'}
        (default 'feynman')

    Returns
    -------
    se : (k,n,n) ndarray
        frequency-dependent self-energy
    '''

    if not _is_tuple(chempot):
        chempot = (chempot, chempot)

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

    nphys = ixq[0].shape[1]
    ndf, nocca, nvira = qja[0].shape
    _, noccb, nvirb = qja[1].shape

    se = np.zeros((grid.shape[0], nphys, nphys), dtype=types.complex128)

    ixq = (ixq[0].reshape((nocca*nphys, ndf)),)
    qja = (qja[0].reshape((ndf, nocca*nvira)), 
           qja[1].reshape((ndf, noccb*nvirb)))

    eova = util.outer_sum([eo[0], -ev[0]]).flatten() - chempot[0]
    eovb = util.outer_sum([eo[1], -ev[1]]).flatten() - chempot[0]

    for i in range(nocca):
        ei_a = eo[0][i] + eova
        ei_b = eo[0][i] + eovb

        xq_a = ixq[0][i*nphys:(i+1)*nphys]

        vi_a = np.dot(xq_a, qja[0]).reshape((nphys, -1))
        vi_b = np.dot(xq_a, qja[1]).reshape((nphys, -1))
        vip_a = np.dot(ixq[0], qja[0][:,i*nvira:(i+1)*nvira])
        vip_a = _reshape_internal(vip_a, (nocca, nphys, nvira), (0,1), (nphys, nocca*nvira))

        di_a = 1.0 / util.outer_sum([w, -ei_a + get_s(ei_a) * grid.eta * 1.0j])
        di_b = 1.0 / util.outer_sum([w, -ei_b + get_s(ei_b) * grid.eta * 1.0j])

        se += util.einsum('wk,xk,yk->wxy', di_a, vi_a, vi_a - vip_a)
        se += util.einsum('wk,xk,yk->wxy', di_b, vi_b, vi_b)

    return se


def build_dfump2_se_direct(e, qpx, qyz, grid, chempot=0.0, ordering='feynman'):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for an unrestricted reference. Poles are summed straight
        into the self-energy and returns an `ndarray` instead of `Aux`.

    Parameters
    ----------
    e : tuple of (n) ndarray
        MO or QMO energies for alpha, beta (beta, alpha) spin
    qpx : 2-tuple of (q,p,x) ndarray
        density-fitted two-electron integrals where first index is in
        the physical basis for alpha, beta (beta, alpha) spin
    qyz : 2-tuple of (q,y,z) ndarray
        density-fitted two-electron integrals for alpha, beta (beta,
        alpha) spin
    grid : (k) ImFqGrid, ImFqQuad or ReFqGrid
        grid
    chempot : tuple of float, optional
        chemical potential for alpha, beta (beta, alpha) spin
    ordering : str
        ordering of the poles {'feynman', 'advanced', 'retarded'}
        (default 'feynman')
    
    Yields
    ------
    se : (k,n,n) ndarray
        frequency-dependent self-energy
    '''

    if not _is_tuple(chempot):
        chempot = (chempot, chempot)

    eo, ev, ixq, qja, axq, qbi = _parse_uhf(e, qpx, qyz, chempot)

    se  = build_dfump2_part_se_direct(eo, ev, ixq, qja, grid, chempot=chempot, ordering=ordering)
    se += build_dfump2_part_se_direct(ev, eo, axq, qbi, grid, chempot=chempot, ordering=ordering)

    return se
