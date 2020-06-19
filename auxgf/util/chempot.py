''' Functions for finding chemical potentials.
'''

import numpy as np


def find_chempot(nphys, nelec, h, occupancy=2.0):
    ''' Finds a chemical potential which best agrees with the number
        of physical electrons using a binary search.

    Parameters
    ----------
    nphys : int
        number of physical degrees of freedom
    nelec : int
        number of physical electrons
    h : (n,n) ndarray
        Hamiltonian describing the physical system, can also be the
        output of util.eigh(h)
    occupancy : int, optional
        occupancy number, i.e. 2 for restricted wavefunctions and
        1 for unrestricted, default 2
    
    Returns
    -------
    chempot : float
        chemical potential
    error : float
        error in the number of electrons
    '''

    if isinstance(h, tuple):
        w, v = h
    else:
        w, v = np.linalg.eigh(h)

    nqmo = v.shape[-1]
    sum_cur = 0.0
    sum_prv = 0.0

    for i in range(nqmo):
        n = occupancy * np.dot(v[:nphys,i].T, v[:nphys,i])
        sum_prv, sum_cur = sum_cur, sum_cur + n

        if i > 0:
            if sum_prv <= nelec and nelec <= sum_cur:
                break

    if abs(sum_prv - nelec) < abs(sum_cur - nelec):
        homo = i-1
        error = nelec - sum_prv
    else:
        homo = i
        error = nelec - sum_cur

    lumo = homo+1
    chempot = 0.5 * (w[lumo] + w[homo])

    return chempot, -error

# Legacy:
_find_chempot = find_chempot
