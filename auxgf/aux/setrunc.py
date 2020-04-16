''' Routines for truncation via self-energy moments.

    A note on QR decomposition methods:

        The slowest step in most AGF2 calculations tends to be the QR
        factorisation of the couplings in the Sigma truncation. For
        this reason I have tried a number of different QR codes, but
        I have settled on a fast Cholesky QR implementation.

        This is unstable in ill-conditioned matrices, however I think
        that for any reasonable AGF2 calculation the couplings should
        be well-conditioned, and if not then there would be other
        problems outside of the QR solver.
'''

import numpy as np
import scipy.linalg
from functools import partial

from auxgf import util
from auxgf.util import types


def _get_qr_function(method='cholesky'):
    if method == 'cholesky':
        qr = util.cholesky_qr
    elif method == 'numpy':
        qr = partial(util.qr, mode='reduced')
    elif method == 'scipy':
        qr = partial(scipy.linalg.qr, mode='economic', 
                     overwrite_a=False, check_finite=False)
    elif method == 'unsafe':
        qr = util.qr_unsafe
    else:
        raise ValueError

    return qr

def build_block_tridiag(m, b):
    ''' Constructs a block tridiagonal matrix from a list of
        on-diagonal and off-diagonal blocks.

    Parameters
    ----------
    m : (k+1,n,n) array
        on-diagonal blocks, the first being h_phys
    b : (k,n,n) array
        off-diagonal blocks

    Returns
    -------
    t : ((k+1)*n,(k+1)*n) ndarray
        block tridiagonal matrix
    '''

    k = len(b)

    if len(m) != k+1:
        raise ValueError('There should be one more on-diagonal block '
                         'than off-diagonal blocks.')

    z = np.zeros(m[0].shape, dtype=types.float64)

    t = np.block([[m[i]   if i == j   else
                   b[j]   if j == i-1 else
                   b[i].T if i == j-1 else z
                   for j in range(k+1)]
                   for i in range(k+1)])

    return t


def block_lanczos(aux, h_phys, nblock, **kwargs):
    ''' Block tridiagonalization of the environment of a Hamiltonian
        spanning the physical and auxiliary space, using the block
        Lanczos algorithm.

    Parameters
    ----------
    aux : Aux
        auxiliaries
    h_phys : (n,n) array
        physical space Hamiltonian
    nblock : int
        number of blocks required = nmom + 1
    debug : bool, optional
        enable debugging tools (default False)
        WARNING: this kills the scaling and memory usage
    reorthog : bool, optional
        enable reorthogonalization of the intermediate Lanczos
        vectors (default True)
    qr: str, optional
        type of QR solver to use {'cholesky', 'numpy', 'scipy', 
        'unsafe'}, default 'cholesky'
    keep_v : bool, optional
        keep and return all of the Lanczos vectors (default False)
        automatically enabled in `debug` is True
        WARNING: this kills the memory usage

    Returns
    -------
    m : (`nblock`+1,n,n) ndarray
        on-diagonal blocks
    b : (`nblock`,n,n) ndarray
        off-diagonal blocks
    v : (`nblock`+1,m,n) ndarray, optional
        Lanczos vectors (only if `keep_v` is True)
    '''

    nphys = aux.nphys
    nqmo = nphys + aux.naux

    keep_v = kwargs.get('debug', False) or kwargs.get('keep_v', False)

    qr = _get_qr_function(method=kwargs.get('qr', 'cholesky'))

    #while nblock >= nqmo // nphys:
    #    nblock -= 1

    v = []
    m = np.zeros((nblock+1, nphys, nphys), dtype=types.float64)
    b = np.zeros((nblock, nphys, nphys), dtype=types.float64)

    v.append(np.eye(nqmo, nphys))
    u = np.ascontiguousarray(np.block([h_phys, aux.v]).T)

    if kwargs.get('debug', False):
        assert np.allclose(np.dot(v[0].T, v[0]), np.eye(nphys))

    for j in range(nblock):
        m[j] = np.dot(u.T, v[-1])
        r = u - np.dot(v[-1], m[j])

        if kwargs.get('reorthog', True):
            r -= np.dot(v[-1], np.dot(v[-1].T, r))

        vnext, b[j] = qr(r)

        if not keep_v:
            v = [v[-1], vnext]
        else:
            v.append(vnext)

        u = aux.dot(h_phys, v[-1])
        u -= np.dot(v[-2], b[j].T)

    m[nblock] = np.dot(u.T, v[-1])

    if kwargs.get('debug', False):
        assert np.allclose(h_phys, m[0])

        vs = np.hstack(v)
        tols = {'rtol': 1e-4, 'atol': 1e-7}

        assert np.allclose(np.dot(vs.T, vs), np.eye(vs.shape[-1]), **tols)

        h = aux.as_hamiltonian(h_phys)
        h_tri = build_block_tridiag(m, b)
        h_proj = util.dots([vs.T, h, vs])

        assert np.allclose(h_proj, h_tri, **tols)

    if not keep_v:
        return m, b
    else:
        return m, b, v


def block_lanczos_1mom(aux, h_phys, **kwargs):
    ''' The above function simplifies significantly in the case of nmom=1.
    '''
    qr = _get_qr_function(method=kwargs.get('qr', 'cholesky'))
    v, b = qr(aux.v.T)
    m = util.einsum('ip,i,iq->pq', v, aux.e, v)
    return [h_phys, m], [b,]


def band_lanczos(aux, h_phys, nblock, **kwargs):
    ''' Band diagonalization of the environment of a Hamiltonain
        spanning the physical and auxiliary space, using the block
        Lanczos algorithm.

    Parameters
    ----------
    aux : Aux
        auxiliaries
    h_phys : (n,n) array
        physical space Hamiltonian
    nblock : int or float
        number of blocks required = nmom + 1, if float is given, the
        size of the resulting space is rounded to an integer
    qr: str, optional
        type of QR solver to use {'cholesky', 'numpy', 'scipy', 
        'unsafe'}, default 'cholesky'
    debug : bool, optional
        enable debugging tools (default False)
        WARNING: this kills the scaling and memory usage

    Returns
    -------
    t : (m,m) ndarray
        Band-diagonal matrix, side length (`nblock`+1)*`nphys`.
    '''

    nphys = aux.nphys
    naux = aux.naux
    nband = int(nblock * nphys)

    qr = _get_qr_function(method=kwargs.get('qr', 'cholesky'))
    v, coup = qr(aux.v.T)

    q = np.zeros((nband, naux), dtype=np.float64)
    q[:min(nphys, naux)] = v.T
    t = np.zeros((nband, nband), dtype=np.float64)

    for i in range(nband):
        r = aux.e * q[i]

        start = max(i-nphys, 0)
        if start != i:
            r -= np.dot(t[i,start:i], q[start:i])

        for j in range(i, min(i+nphys, nband)):
            t[i,j] = t[j,i] = scipy.linalg.blas.ddot(r, q[j])
            scipy.linalg.blas.daxpy(q[j], r, a=-t[i,j])

        if i+nphys < nband:
            norm_r = util.norm(r)
            t[i,i+nphys] = t[i+nphys,i] = norm_r
            q[i+nphys] = r / (norm_r + 1e-100)
    
    if kwargs.get('debug', False):
        # This might fail if naux < nphys
        assert np.allclose(np.dot(q, aux.e[:,None] * q.T), t)

    coup_block = np.zeros((nband, nphys), dtype=np.float64)
    coup_block[:min(nphys, naux)] = coup

    t = np.block([[h_phys, coup_block.T], [coup_block, t]])

    return t


def build_auxiliaries(h, nphys):
    ''' Builds a set of auxiliary energies and couplings from the
        block tridiagonal Hamiltonian.

    Parameters
    ----------
    h : (n,n) ndarray
        block tridiagonal Hamiltonian
    nphys : int
        number of physical degrees of freedom

    Returns
    -------
    e : (m) ndarray
        auxiliary energies
    v : (nphys,m) ndarray
        auxiliary couplings
    '''

    w, v = util.eigh(h[nphys:,nphys:])

    e = w

    v = np.dot(h[:nphys,nphys:2*nphys], v[:nphys])

    return e, v


def run(aux, h_phys, nmom, method='band', qr='cholesky'):
    ''' Runs the truncation by moments of the self-energy.
        
        [1] H. Muther, T. Taigel and T. T. S. Kuo, Nucl. Phys., 482, 
            1988, pp. 601-616.
        [2] D. Van Neck, K. Peirs and M. Waroquier, J. Chem. Phys.,
            115, 2001, pp. 15-25.
        [3] H. Muther and L. D. Skouras, Nucl. Phys., 555, 1993, 
            pp. 541-562.
        [4] Y. Dewulf, D. Van Neck, L. Van Daele and M. Waroquier,
            Phys. Lett. B, 396, 1997, pp. 7-14.

    Parameters
    ----------
    aux : Aux
        auxiliaries
    h_phys : (n,n) ndarray
        physical space Hamiltonian
    nmom : int
        maximum moment order
    method : str, optional
        type of diagonalizer to use {'block', 'band'}, default 'band'
    qr: str, optional
        type of QR solver to use {'cholesky', 'numpy', 'scipy', 
        'unsafe'}, default 'cholesky'

    Returns
    -------
    red : Aux
        reduced auxiliaries
    '''

    #TODO: debugging mode which checks the moments

    if nmom == 0:
        return aux.new([], [[],]*aux.nphys)

    elif nmom == 1:
        m_occ, b_occ = block_lanczos_1mom(aux.as_occupied(), 
                                          h_phys, qr=qr)
        m_vir, b_vir = block_lanczos_1mom(aux.as_virtual(), 
                                          h_phys, qr=qr)

        t_occ = build_block_tridiag(m_occ, b_occ)
        t_vir = build_block_tridiag(m_vir, b_vir)

    elif method == 'block':
        m_occ, b_occ = block_lanczos(aux.as_occupied(), 
                                     h_phys, nmom, qr=qr)
        m_vir, b_vir = block_lanczos(aux.as_virtual(),
                                     h_phys, nmom, qr=qr)

        t_occ = build_block_tridiag(m_occ, b_occ)
        t_vir = build_block_tridiag(m_vir, b_vir)

    else:
        t_occ = band_lanczos(aux.as_occupied(), 
                             h_phys, nmom, qr=qr)
        t_vir = band_lanczos(aux.as_virtual(),
                             h_phys, nmom, qr=qr)

    e_occ, v_occ = build_auxiliaries(t_occ, aux.nphys)
    e_vir, v_vir = build_auxiliaries(t_vir, aux.nphys)

    red_occ = aux.new(e_occ, v_occ)
    red_vir = aux.new(e_vir, v_vir)
    red = red_occ + red_vir

    return red



