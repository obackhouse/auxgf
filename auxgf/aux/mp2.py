''' Routines for building sets of MP2 auxiliaries.
'''

import numpy as np


from auxgf.aux.rmp2 import *
from auxgf.aux.ump2 import *
from auxgf.aux.dfrmp2 import *
from auxgf.aux.dfump2 import *
from auxgf import util


def build_mp2_part(eo, ev, xija, **kwargs):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams.

    Parameters
    ----------
    See either auxgf.aux.build_rmp2.build_rmp2_part or
    auxgf.aux.build_ump2.build_ump2_part

    Returns
    -------
    e : (m) ndarray
        auxiliary energies
    v : (n,m) ndarray
        auxiliary couplings
    '''

    ndim = util.iter_depth(eo)

    if ndim == 1:
        return build_rmp2_part(eo, ev, xija, **kwargs)
    elif ndim == 2:
        return build_ump2_part(eo, ev, xija, **kwargs)
    else:
        raise ValueError


def build_mp2(e, eri, chempot=0.0, **kwargs):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams. Output is controlled by input dimensions:

        If e is (n,) and eri (n,m,m,m):
            restricted, return single `Aux`
        If e is (2,n) and eri (2,n,m,m,m)
            unrestricted, return single `Aux`
        If e is (2,n) and eri (2,2,n,m,m,m)
            unrestricted, return both `Aux`

    Parameters
    ----------
    See either auxgf.aux.build_rmp2.build_rmp2 or 
    auxgf.aux.build_ump2.build_ump2

    Returns
    -------
    poles : Aux
        auxiliaries
    '''

    ndim = util.iter_depth(e)

    if ndim == 1:
        return build_rmp2(e, eri, chempot=chempot, **kwargs)
    elif ndim == 2:
        eri_ndim = util.iter_depth(eri)

        if eri_ndim == 5:
            return build_ump2(e, eri, chempot=chempot, **kwargs)
        elif eri_ndim == 6:
            a = build_ump2(e, eri[0], chempot=chempot, **kwargs)
            b = build_ump2(e[::-1], eri[1][::-1], chempot=chempot[::-1], **kwargs)
            return a, b
        else:
            raise ValueError
    else:
        raise ValueError


def build_mp2_iter(se, h_phys, eri_mo, **kwargs):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams by iterating the current set of auxiliaries according
        to the eigenvalue form of the Dyson equation.

    Parameters
    ----------
    se : Aux
        auxiliaries of previous iteration
    eri_mo : (n,n,n,n) ndarray
        two-electron repulsion integrals in MO basis
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

    ndim = util.iter_depth(se)

    if ndim == 0:
        return build_rmp2_iter(se, h_phys, eri_mo, **kwargs)
    elif ndim == 1:
        return build_ump2_iter(se, h_phys, eri_mo, **kwargs)
    else:
        raise ValueError


def build_dfmp2_part(eo, ev, ixq, qja, **kwargs):
    ''' Builds a set of auxiliaries representing all (i,j,a) or (a,b,i)
        diagrams.

    Parameters
    ----------
    See either auxgf.aux.build_rmp2.build_rmp2_part or
    auxgf.aux.build_ump2.build_ump2_part

    Returns
    -------
    e : (m) ndarray
        auxiliary energies
    v : (n,m) ndarray
        auxiliary couplings
    '''

    ndim = util.iter_depth(eo)

    if ndim == 1:
        return build_dfrmp2_part(eo, ev, ixq, qja, **kwargs)
    elif ndim == 2:
        return build_dfump2_part(eo, ev, ixq, qja, **kwargs)
    else:
        raise ValueError


def build_dfmp2(e, qpx, qyz, chempot=0.0, **kwargs):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams. Output is controlled by input dimensions:

        If e is (n,) and qpx (q,n,m):
            restricted, return single `Aux`
        If e is (2,n) and qpx (2,q,n,m):
            unrestricted, return both `Aux`

    Parameters
    ----------
    See either auxgf.aux.build_rmp2.build_rmp2 or 
    auxgf.aux.build_ump2.build_ump2

    Returns
    -------
    poles : Aux
        auxiliaries
    '''

    ndim = util.iter_depth(e)

    if ndim == 1:
        return build_dfrmp2(e, qpx, qyz, chempot=chempot, **kwargs)
    elif ndim == 2:
        a = build_dfump2(e, qpx, qyz, chempot=chempot, **kwargs)
        b = build_dfump2(e[::-1], qpx[::-1], qyz[::-1], chempot=chempot[::-1], **kwargs)
        return a, b
    else:
        raise ValueError


def build_dfmp2_iter(se, h_phys, eri_mo, **kwargs):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams by iterating the current set of auxiliaries according
        to the eigenvalue form of the Dyson equation.

    Parameters
    ----------
    se : Aux
        auxiliaries of previous iteration
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
    poles : Aux
        auxiliaries
    '''

    ndim = util.iter_depth(se)

    if ndim == 0:
        return build_dfrmp2_iter(se, h_phys, eri_mo, **kwargs)
    elif ndim == 1:
        return build_dfump2_iter(se, h_phys, eri_mo, **kwargs)
    else:
        raise ValueError
