''' Routines for building sets of auxiliaries.
'''

import numpy as np


from auxgf.aux.build_rmp2 import *
from auxgf.aux.build_ump2 import *


def build_mp2_part(eo, ev, xija, wtol=1e-10):
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

    if np.asarray(eo).ndim == 1:
        return build_rmp2_part(eo, ev, xija, wtol=wtol)
    else:
        return build_ump2_part(eo, ev, xija, wtol=wtol)


def build_mp2(e, eri, chempot=0.0, wtol=1e-10):
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

    if np.asarray(e).ndim == 1:
        return build_rmp2(e, eri, chempot=chempot, wtol=wtol)
    else:
        if np.asarray(eri).ndim == 5:
            return build_ump2(e, eri, chempot=chempot, wtol=wtol)
        elif np.asarray(eri).ndim == 6:
            a = build_ump2(e, eri[0], chempot=chempot, wtol=wtol)
            b = build_ump2(e[::-1], eri[1][::-1], chempot=chempot, wtol=wtol)
            return a, b


def build_mp2_iter(aux, h_phys, eri_mo, wtol=1e-10):
    ''' Builds a set of auxiliaries representing all (i,j,a) and (a,b,i)
        diagrams by iterating the current set of auxiliaries according
        to the eigenvalue form of the Dyson equation.

    Parameters
    ----------
    aux : Aux
        auxiliaries of previous iteration
    eri_mo : (n,n,n,n) ndarray
        two-electron repulsion integrals in MO basis
    wtol : float, optional
        threshold for an eigenvalue to be considered zero

    Returns
    -------
    poles : Aux
        auxiliaries
    '''

    if np.asarray(e).ndim == 1:
        return build_rmp2_iter(aux, h_phys, eri_mo, wtol=wtol)
    else:
        return build_ump2_iter(aux, h_phys, eri_mo, wtol=wtol)















