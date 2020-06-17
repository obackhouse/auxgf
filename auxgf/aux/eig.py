''' Routines for diagonalising auxiliary + physical spaces.
'''

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

from auxgf import aux, util
from auxgf.util import log


def _check_dims(se, h_phys):
    nphys, naux = se.v.shape
    assert h_phys.shape == (nphys, nphys)


def eigh(se, h_phys, **kwargs):
    if kwargs.get('nroots', None) in [None, -1]:
        return dense(se, h_phys, chempot=kwargs.get('chempot', 0.0))
    else:
        return davidson(se, h_phys, **kwargs)


def dense(se, h_phys, chempot=0.0):
    ''' Diagonalise the auxiliary + physical space via a dense
        eigensolver.
    '''

    _check_dims(se, h_phys)

    h_ext = se.as_hamiltonian(h_phys, chempot=chempot)
    w, v = np.linalg.eigh(h_ext)

    return w, v


def davidson(se, h_phys, chempot=0.0, nroots=1, which='SM', tol=1e-14, maxiter=None, ntrial=None):
    ''' Diagonalise the auxiliary + physical space via a
        Davidson iterative eigensolver.
    '''

    _check_dims(se, h_phys)
    ndim = se.nphys + se.naux

    if maxiter is None: maxiter = 10 * ndim
    if ntrial is None: ntrial = min(ndim, max(2*nroots+1, 20))

    abs_op = np.absolute if which in ['SM', 'LM'] else lambda x: x
    order = 1 if which in ['SM', 'SA'] else -1

    matvec = lambda x : se.dot(h_phys, np.asarray(x))

    diag = np.concatenate([np.diag(h_phys), se.e])

    guess = [np.zeros((ndim)) for n in range(nroots)]
    mask = np.argsort(abs_op(diag))[::order]
    for i in range(nroots):
        guess[i][mask[i]] = 1

    def pick(w, v, nroots, callback):
        mask = np.argsort(abs_op(w))
        mask = mask[::order]
        w = w[mask]
        v = v[:,mask]
        return w, v, 0

    conv, w, v = util.davidson(matvec, guess, diag, tol=tol, nroots=nroots, 
                               max_space=ntrial, max_cycle=maxiter, pick=pick)

    if not conv:
        log.warn('Davidson solver did not converge.')

    return w, v


def lanczos(se, h_phys, chempot=0.0, nroots=1, which='SM', tol=1e-14, maxiter=None, ntrial=None):
    ''' Diagonalise the auxiliary + physical space via a 
        Lanczos iterative eigensolver.
    '''

    _check_dims(se, h_phys)
    ndim = se.nphys + se.naux

    if maxiter is None: maxiter = 10 * ndim
    if ntrial is None: ntrial = min(ndim, max(2*nroots+1, 20))

    matvec = lambda x : se.dot(h_phys, np.asarray(x))

    linop = LinearOperator(shape=(ndim, ndim), dtype=np.float64, matvec=matvec)

    w, v = eigsh(linop, k=nroots, which=which, tol=tol, ncv=ntrial, maxiter=maxiter)

    return w, v
