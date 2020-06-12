''' Routines for building sets of auxiliaries according to the UMP2
    self-energy.
'''

import numpy as np

from auxgf import util, aux
from auxgf.util import types


_is_tuple = lambda x : isinstance(x, (tuple, list, np.ndarray))


def _parse_uhf(e, eri, chempot):
    if not (np.all(np.diff(e[0]) >= 0) and np.all(np.diff(e[1])) >= 0):
        # See auxgf.aux.build.rmp2._parse_rhf

        oa = e[0] < chempot[0]
        va = e[0] >= chempot[0]
        ob = e[1] < chempot[1]
        vb = e[1] >= chempot[1]

        eo = (e[0][oa], e[1][ob])
        ev = (e[0][va], e[1][vb])

        xija_aaaa = eri[0][:,:,:,va][:,:,oa][:,oa]
        xija_aabb = eri[1][:,:,:,vb][:,:,ob][:,oa]
        xabi_aaaa = eri[0][:,:,:,oa][:,:,va][:,va]
        xabi_aabb = eri[1][:,:,:,ob][:,:,vb][:,va]

    else:
        oa = slice(None, np.sum(e[0] < chempot[0]))
        va = slice(np.sum(e[0] < chempot[0]), None)
        ob = slice(None, np.sum(e[1] < chempot[1]))
        vb = slice(np.sum(e[1] < chempot[1]), None)

        eo = (e[0][oa], e[1][ob])
        ev = (e[0][va], e[1][vb])

        xija_aaaa = eri[0][:,oa,oa,va]
        xija_aabb = eri[1][:,oa,ob,vb]
        xabi_aaaa = eri[0][:,va,va,oa]
        xabi_aabb = eri[1][:,va,vb,ob]

    xija = (xija_aaaa, xija_aabb)
    xabi = (xabi_aaaa, xabi_aabb)

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


def build_ump2_part(eo, ev, xija, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for an unrestricted reference.

    Parameters
    ----------
    eo : 2-tuple of (o) ndarray
        occupied (virtual) energies for alpha, beta (beta, alpha) spin
    ev : 2-tuple of (v) ndarray
        virtual (occupied) energies for alpha, beta (beta, alpha) spin
    xija : 2-tuple of (n,o,o,v)
        two-electron integrals indexed as physical, occupied, occupied,
        virtual (physical, virtual, virtual, occupied) for (aa|aa),
        (aa|bb) [(bb|bb), (bb|aa)] spin
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

    nphys, nocca, _, nvira = xija[0].shape
    _, _, noccb, nvirb = xija[1].shape
    npoles  = nvira * nocca * (nocca-1) // 2
    npoles += nvirb * nocca * noccb

    e = np.zeros((npoles), dtype=types.float64)
    v = np.zeros((nphys, npoles), dtype=types.float64)

    a_factor = np.sqrt(ss_factor)
    b_factor = np.sqrt(os_factor)

    n0 = 0
    for i in range(nocca):
        nja_a = i * nvira
        nja_b = noccb * nvirb
        jm = slice(None, i)
        am = slice(n0, n0+nja_a)
        bm = slice(n0+nja_a, n0+nja_a+nja_b)

        vija_aaa = xija[0][:,i,jm].reshape((nphys, -1))
        vjia_aaa = xija[0][:,jm,i].reshape((nphys, -1))
        vija_abb = xija[1][:,i].reshape((nphys, -1))

        e[am] = eo[0][i] + np.subtract.outer(eo[0][jm], ev[0]).flatten()
        e[bm] = eo[0][i] + np.subtract.outer(eo[1], ev[1]).flatten()

        # FIXME: is this line correct? originally I did not have the brackets here
        v[:,am] = a_factor * (vija_aaa - vjia_aaa)
        v[:,bm] = b_factor * vija_abb

        n0 += nja_a + nja_b

    mask = np.sum(v*v, axis=0) >= wtol
    e = e[mask]
    v = v[:,mask]

    e = e[:n0]
    v = v[:,:n0]

    assert e.shape[0] == v.shape[1]

    return e, v


def build_ump2(e, eri, chempot=0.0, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for an unrestricted reference.

    Parameters
    ----------
    e : tuple of (n) ndarray
        MO or QMO energies for alpha, beta (beta, alpha) spin
    eri : tuple of (n,m,m,m)
        two-electron integrals where the first index is in the physical
        basis for (aa|aa), (aa|bb) [(bb|bb), (bb|aa)] spin
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

    assert np.asarray(e).ndim == 2
    assert np.asarray(eri).ndim == 5

    eo, ev, xija, xabi = _parse_uhf(e, eri, chempot)

    eija, vija = build_ump2_part(eo, ev, xija, wtol=wtol,
                                 ss_factor=ss_factor, os_factor=os_factor)
    eabi, vabi = build_ump2_part(ev, eo, xabi, wtol=wtol,
                                 ss_factor=ss_factor, os_factor=os_factor)

    e = np.concatenate((eija, eabi), axis=0)
    v = np.concatenate((vija, vabi), axis=1)

    poles = aux.Aux(e, v, chempot=chempot[0])

    return poles


def build_ump2_iter(aux, h_phys, eri_mo, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
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
    eri_mo : (2,2,n,n,n,n) ndarray
        two-electron repulsion integrals in MO basis, where first two
        indices are spin indices for the bra and ket
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

    ea, ca = aux[0].eig(h_phys[0])
    eb, cb = aux[1].eig(h_phys[1])

    oa = ea < aux[0].chempot
    ob = eb < aux[1].chempot
    va = ea >= aux[0].chempot
    vb = eb >= aux[1].chempot

    eo = (ea[oa], eb[ob])
    ev = (ea[va], eb[vb])
    co = (ca[:aux[0].nphys,oa], cb[:aux[0].nphys,ob])
    cv = (ca[:aux[1].nphys,va], cb[:aux[1].nphys,vb])

    xija_aaaa = util.mo2qo(eri_mo[0,0], co[0], co[0], cv[0])
    xija_aabb = util.mo2qo(eri_mo[0,1], co[0], co[1], cv[1])
    xija = (xija_aaaa, xija_aabb)
    eija_a, vija_a = build_ump2_part(eo, ev, xija, wtol=wtol,
                                     ss_factor=ss_factor, os_factor=os_factor)
    del xija

    xija_bbbb = util.mo2qo(eri_mo[1,1], co[1], co[1], cv[1])
    xija_bbaa = util.mo2qo(eri_mo[1,0], co[1], co[0], cv[0])
    xija = (xija_bbbb, xija_bbaa)
    eija_b, vija_b = build_ump2_part(eo[::-1], ev[::-1], xija, wtol=wtol,
                                     ss_factor=ss_factor, os_factor=os_factor)
    del xija

    xabi_aaaa = util.mo2qo(eri_mo[0,0], cv[0], cv[0], co[0])
    xabi_aabb = util.mo2qo(eri_mo[0,1], cv[0], cv[1], co[1])
    xabi = (xabi_aaaa, xabi_aabb)
    eabi_a, vabi_a = build_ump2_part(ev, eo, xabi, wtol=wtol,
                                     ss_factor=ss_factor, os_factor=os_factor)
    del xabi

    xabi_bbbb = util.mo2qo(eri_mo[1,1], cv[1], cv[1], co[1])
    xabi_bbaa = util.mo2qo(eri_mo[1,0], cv[1], cv[0], co[0])
    xabi = (xabi_bbbb, xabi_bbaa)
    eabi_b, vabi_b = build_ump2_part(ev[::-1], eo[::-1], xabi, wtol=wtol,
                                     ss_factor=ss_factor, os_factor=os_factor)
    del xabi

    ea = np.concatenate((eija_a, eabi_a), axis=0)
    eb = np.concatenate((eija_b, eabi_b), axis=0)
    va = np.concatenate((vija_a, vabi_a), axis=1)
    vb = np.concatenate((vija_b, vabi_b), axis=1)

    poles_a = aux[0].new(ea, va)
    poles_b = aux[1].new(eb, vb)

    return poles_a, poles_b


def build_ump2_part_direct(eo, ev, xija, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for an unrestricted reference. Uses a generator which
        iterates over blocks.

    Parameters
    ----------
    eo : 2-tuple of (o) ndarray
        occupied (virtual) energies for alpha, beta (beta, alpha) spin
    ev : 2-tuple of (v) ndarray
        virtual (occupied) energies for alpha, beta (beta, alpha) spin
    xija : 2-tuple of (n,o,o,v)
        two-electron integrals indexed as physical, occupied, occupied,
        virtual (physical, virtual, virtual, occupied) for (aa|aa),
        (aa|bb) [(bb|bb), (bb|aa)] spin
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

    nphys, nocca, _, nvira = xija[0].shape
    _, _, noccb, nvirb = xija[1].shape
    npoles  = nvira * nocca * (nocca-1) // 2
    npoles += nvirb * nocca * noccb

    a_factor = np.sqrt(ss_factor)
    b_factor = np.sqrt(os_factor)

    for i in range(nocca):
        nja_a = i * nvira
        nja_b = noccb * nvirb
        jm = slice(None, i)

        vija_aaa = xija[0][:,i,jm].reshape((nphys, -1))
        vjia_aaa = xija[0][:,jm,i].reshape((nphys, -1))
        vija_abb = xija[1][:,i].reshape((nphys, -1))

        ea = eo[0][i] + np.subtract.outer(eo[0][jm], ev[0]).flatten()
        eb = eo[0][i] + np.subtract.outer(eo[1], ev[1]).flatten()

        # FIXME: is this line correct? originally I did not have the brackets here
        va = a_factor * (vija_aaa - vjia_aaa)
        vb = b_factor * vija_abb

        if len(ea):
            yield ea, va
        if len(eb):
            yield eb, vb


def build_ump2_direct(e, eri, chempot=0.0, wtol=1e-12, ss_factor=1.0, os_factor=1.0):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for an unrestricted reference. Uses a generator which
        iterates over blocks.

    Parameters
    ----------
    e : tuple of (n) ndarray
        MO or QMO energies for alpha, beta (beta, alpha) spin
    eri : tuple of (n,m,m,m)
        two-electron integrals where the first index is in the physical
        basis for (aa|aa), (aa|bb) [(bb|bb), (bb|aa)] spin
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

    assert _is_tuple(e) and len(e) == 2
    assert _is_tuple(eri) and len(eri) == 2
    
    eo, ev, xija, xabi = _parse_uhf(e, eri, chempot)

    kwargs = dict(ss_factor=ss_factor, os_factor=os_factor, wtol=wtol)

    for e,v in build_ump2_part_direct(eo, ev, xija, **kwargs): 
        yield aux.Aux(e, v, chempot=chempot[0])

    for e,v in build_ump2_part_direct(ev, eo, xabi, **kwargs): 
        yield aux.Aux(e, v, chempot=chempot[1])


def build_ump2_part_se_direct(eo, ev, xija, grid, chempot=0.0, ordering='feynman'):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams for an unrestricted reference. Poles are summed straight
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

    w_a = grid.prefac * grid.values
    w_b = grid.prefac * grid.values

    nphys, nocca, _, nvira = xija[0].shape
    se = np.zeros((grid.shape[0], nphys, nphys), dtype=types.complex128)

    eova = util.outer_sum([eo[0], -ev[0]]).flatten() - chempot[0]
    eovb = util.outer_sum([eo[1], -ev[1]]).flatten() - chempot[0]

    for i in range(nocca):
        ei_a = eo[0][i] + eova
        ei_b = eo[0][i] + eovb

        vi_a = xija[0][:,i].reshape((nphys, -1))
        vip_a = xija[0][:,:,i].reshape((nphys, -1))
        vi_b = xija[1][:,i].reshape((nphys, -1))

        di_a = 1.0 / util.outer_sum([w_a, -ei_a + get_s(ei_a) * grid.eta * 1.0j])
        di_b = 1.0 / util.outer_sum([w_b, -ei_b + get_s(ei_b) * grid.eta * 1.0j])

        se += util.einsum('wk,xk,yk->wxy', di_a, vi_a, vi_a - vip_a)
        se += util.einsum('wk,xk,yk->wxy', di_b, vi_b, vi_b)

    return se


def build_ump2_se_direct(e, eri, grid, chempot=0.0, ordering='feynman'):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams for an unrestricted reference. Poles are summed straight
        into the self-energy and returns an `ndarray` instead of `Aux`.

    Parameters
    ----------
    e : tuple of (n) ndarray
        MO or QMO energies for alpha, beta (beta, alpha) spin
    eri : tuple of (n,m,m,m)
        two-electron integrals where the first index is in the physical
        basis for (aa|aa), (aa|bb) [(bb|bb), (bb|aa)] spin
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

    eo, ev, xija, xabi = _parse_uhf(e, eri, chempot)

    se  = build_ump2_part_se_direct(eo, ev, xija, grid, chempot=chempot)
    se += build_ump2_part_se_direct(ev, eo, xabi, grid, chempot=chempot)

    return se
