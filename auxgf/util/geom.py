import numpy as np

from auxgf.util import types


def ring(label, num, d):
    ''' Builds a ring of a given atom using pyscf.tools.ring

    Parameters
    ----------
    label : str
        atom label
    num : int
        number of atoms in the ring
    d : float
        distance between atoms in the ring

    Returns
    -------
    atoms
        a list of tuples of (label, np.array([x, y, z])) for each atom
    '''

    from pyscf.tools import ring as _ring

    atoms = [(label, np.asarray(x, dtype=types.float64)) 
             for x in _ring.make(num, d)]

    return atoms


def tetrahedron(labels, d):
    ''' Builds a tetrahedral molecule from a set of labels. The first
        label corresponds to the central atom, and the subsequent ones
        ordered according to the R-stereoisomer.

        i.e. `labels = ['C', 'H', 'H', 'H']` for methane

    Parameters
    ----------
    label : list of str
        atom labels
    d : float
        distance between central atom and others

    Returns
    -------
    atoms
        a list of tuples of (label, np.array([x, y, z])) for each atom
    '''

    coords = [ 
        np.asarray([ 0.0,  0.0,  0.0], dtype=types.float64),
        np.asarray([ 0.5,  0.5, -0.5], dtype=types.float64),
        np.asarray([ 0.5, -0.5,  0.5], dtype=types.float64),
        np.asarray([-0.5, -0.5, -0.5], dtype=types.float64),
        np.asarray([-0.5,  0.5,  0.5], dtype=types.float64)
    ]

    for i in range(1, 4):
        coords[i] *= (d / np.linalg.norm(coords[i]))

    atoms = list(zip(labels, coords))

    return atoms

