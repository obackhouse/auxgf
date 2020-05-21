''' Functions for finding chemical potentials.
'''

import numpy as np


def find_chempot(nphys, nelec, h, occupancy=2.0):
    #FIXME: this might use a binary search but it's still O(n^2) :D
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

    nelecs = np.einsum('ij,ij->j', v[:nphys,:], v[:nphys,:])
    nelecs *= occupancy

    sums = np.cumsum(nelecs)

    i = np.searchsorted(sums, nelec)

    error_up = sums[i] - nelec
    error_dn = sums[i-1] - nelec

    if abs(error_up) < abs(error_dn):
        homo = i
        lumo = i+1
        error = error_up
    else:
        homo = i-1
        lumo = i
        error = error_dn

    chempot = 0.5 * (w[lumo] + w[homo])

    return chempot, error

def _find_chempot(nphys, nelec, h, occupancy=2.0):
    if isinstance(h, tuple):
        w, v = h
    else:
        w, v = np.linalg.eigh(h)

    nqmo = v.shape[-1]
    nelecs = np.zeros((nqmo,))
    sums = np.zeros((nqmo,))

    for i in range(nqmo):
        nelecs[i] = occupancy * np.dot(v[:nphys,i].T, v[:nphys,i])
        sums[i] = sums[i-1] + nelecs[i]

        if i > 0:
            if sums[i-1] <= nelec and nelec <= sums[i]:
                break

    homo = i-1 if abs(sums[i-1] - nelec) < abs(sums[i] - nelec) else i
    lumo = homo + 1
    error = abs(sums[homo] - nelec)
    chempot = 0.5 * (w[lumo] + w[homo])

    return chempot, error

