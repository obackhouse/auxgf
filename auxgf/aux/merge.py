''' Routines to merge the auxiliaries.
'''

import numpy as np

from auxgf import util
from auxgf.util import types


def aux_merge_exact(aux, etol=1e-10, wtol=1e-10):
    ''' Performs an in-principle exact reduction of the auxiliaries
        which have linear dependencies or negligible weight.

    Parameters
    ----------
    aux : Aux
        auxiliaries
    etol : float, optional
        maximum difference in degenerate energies (default 1e-10)
    wtol : float, optional
        maximum weight to be considered negligible (default 1e-10)

    Returns
    -------
    red : Aux
        reduced auxiliaries
    '''

    #TODO: I think there's still too much python iteration here
    # for this to be optimal

    if etol == 0:
        e_round = aux.e
    else:
        e_round = np.around(aux.e, decimals=-int(np.log10(etol)))

    e_uniq, inds, cts = np.unique(e_round, return_index=True, 
                                  return_counts=True)

    if e_uniq.shape == (aux.naux,):
        return aux

    inds_uniq = inds[cts == 1]
    inds_degen = inds[cts > 1]
    cts_degen = cts[cts > 1]

    e_a = aux.e[inds_uniq]
    v_a = aux.v[:,inds_uniq]

    mask = np.einsum('xk,xk->k', v_a, v_a) > wtol
    e_a = e_a[mask]
    v_a = v_a[:,mask]

    v_list = [aux.v[:,i:i+c] for i,c in zip(inds_degen, cts_degen)]

    m = [np.dot(x.T, x) for x in v_list]

    w, l = util.batch_eigh(m)
    w, l = zip(*[(x[x >= wtol], y[:,x >= wtol]) for x,y in zip(w, l)])
    
    l = [np.dot(a, b) for a,b in zip(v_list, l)]

    e_b = np.repeat(aux.e[inds_degen], [x.shape[1] for x in l])

    l = np.concatenate(l, axis=1)
    w = np.concatenate(w, axis=0)

    v_b = l
    v_b = util.normalise(v_b, shift=0)
    v_b *= np.sqrt(w)

    assert v_a.shape == (aux.nphys, e_a.shape[0])
    assert v_b.shape == (aux.nphys, e_b.shape[0])

    e = np.concatenate((e_a, e_b), axis=0)
    v = np.concatenate((v_a, v_b), axis=1)

    red = aux.new(e, v)

    return red


