''' Linear algebra and extensions to numpy.
'''

import numpy as np
import functools
from pyscf.lib import linalg_helper
from pyscf.lib import einsum as pyscf_einsum
from pyscf.lib import direct_sum as pyscf_dirsum
from scipy.linalg import blas
import os

try:
    from tblis_einsum import einsum as tblis_einsum
except ImportError:
    tblis_einsum = None

from auxgf.util import types, mkl, log, mpi

einsum = 'tblis'  # 'numpy', 'pyscf', 'tblis'


''' Wrapper for einsum: choose between implementations, and clear up
    `optimize` keyword inconsistencies between different versions of
    numpy.
'''

numpy_einsum = functools.partial(np.einsum, casting='safe',
                                 order='C', optimize=True)

pyscf_einsum = functools.partial(pyscf_einsum, casting='safe',
                                 order='C', optimize=True)

if tblis_einsum is not None:
    tblis_einsum = functools.partial(tblis_einsum, casting='safe',
                                     order='C', optimize=True)

def _tblis_einsum(key, *args, **kwargs):
    # Does tblis_einsum not support Ellipsis in the key?
    if '...' in key or tblis_einsum is None:
        return numpy_einsum(key, *args, **kwargs)
    else:
        return tblis_einsum(key, *args, **kwargs)

if einsum == 'numpy':
    einsum = numpy_einsum
elif einsum == 'pyscf':
    einsum = pyscf_einsum
elif einsum == 'tblis':
    einsum = _tblis_einsum
else:
    raise ValueError


''' Wrapper for np.linalg to avoid the long names.
'''

norm = np.linalg.norm
eigh = np.linalg.eigh
dots = np.linalg.multi_dot
qr = np.linalg.qr
eigvalsh = np.linalg.eigvalsh


def _reorder_fortran(a, trans_a=False):
    ''' Reorders array into Fortran memory contiguity.
    '''

    if a.flags.c_contiguous:
        return np.array(a.T, copy=False, order='F'), not trans_a
    else:
        return np.array(a, copy=False, order='F'), trans_a

def _reorder_c(a, trans_a=False):
    ''' Reorders array into C memory contiguity.
    '''

    if a.flags.f_contiguous:
        return np.array(a.T, copy=False, order='C'), not trans_a
    else:
        return np.array(a, copy=False, order='C'), trans_a

_is_contiguous = lambda a: a.flags.c_contiguous or a.flags.f_contiguous


def dgemm(a, b, c=None, alpha=1.0, beta=0.0):
    ''' Performs dgemm in Fortran memory alignment without copying.
        Input matrix should be contiguous (either F or C).
    
    Parameters
    ----------
    a : array
        input matrix a
    b : array
        input matrix b
    c : array, optional
        output matrix c, if None then it is allocated inside the
        function, default None
    alpha : float, optional
        scalar factor for matrix a
    beta : float, optional
        scalar factor for matrix c

    Returns
    -------
    c : ndarray
        output matrix
    '''

    #FIXME: this needs testing better - currently this should only be
    # used where it has been tested against np.dot specifically for 
    # that use case!

    if (not _is_contiguous(a)) or (not _is_contiguous(b)):
        log.warn('DGEMM called on non-contiguous data')

    m, k = a.shape
    n = b.shape[1]
    assert k == b.shape[0]

    a, ta = _reorder_fortran(a)
    b, tb = _reorder_fortran(b)

    if c is None:
        c = np.zeros((m, n), dtype=np.float64, order='C')

    if m == 0 or n == 0 or k == 0:
        return c

    c, tc = _reorder_fortran(c)

    c = blas.dgemm(alpha=alpha, a=b, b=a, c=c, beta=beta, 
                   trans_a=not tb, trans_b=not ta)

    c, tc = _reorder_c(c)

    return c


def qr_unsafe(a): # pragma: no cover
    ''' Performs a QR decomposition of an array. This is faster than
        `numpy.linalg.qr` for C-contiguous arrays.

        If `a.flags['C_CONTIGUOUS']`, `a` is overwritten and `q` will
        use the same memory. This does not apply if `a` is not an
        ndarray or has `a.flags['F_CONTIGUOUS']`.

        Equivalent to `np.linalg.qr(a, mode='reduced')`.

    Parameters
    ----------
    a : (m,n) array
        array to be orthogonalised

    Returns
    -------
    q : (m,n) ndarray
        orthogonalised vectors
    r : (n,n) ndarray
        upper-triangular matrix
    '''

    # I have seen some bugs with this so I am tentative to use it
    raise NotImplementedError

    if not mkl.has_mkl:
        return np.linalg.qr(a)

    a = np.ascontiguousarray(a)

    m, n = a.shape
    mn = min(m, n)
    tau = np.zeros((max(1, mn),), dtype=a.dtype)

    mkl.dgeqrf('c', m, n, a, max(1, n), tau)

    q = a.copy()

    mkl.dorgqr('c', m, mn, mn, q, max(1, n), tau)

    q = q[:,:mn]
    r = np.triu(a[:mn])

    return q, r


def cholesky_qr(a):
    ''' Performs the Cholesky QR decomposition. This can be unstable.
    '''

    try:
        x = np.dot(a.T, a)
        r = np.linalg.cholesky(x).T
        q = np.dot(a, np.linalg.inv(r))

    except np.linalg.LinAlgError: # pragma: no cover
        if mpi.rank:
            log.warn('Matrix not positive definite in Cholesky step of '
                     'util.linalg.qholesky_qr - falling back to numpy.')
        return np.linalg.qr(a)

    return q, r


def cholesky_qr2(a):
    ''' Refines the instability of cholesky_qr.
    '''

    q1, r1 = cholesky_qr(a)
    q2, r2 = cholesky_qr(q1)
    r2 = np.dot(r2, r1)

    return q2, r2


def block_diag(arrays):
    ''' Constructs a block diagonal array from a series of arrays.
        Input arrays don't need to be square or the same shape.

    Parameters
    ----------
    arrays : list of ndarrays
        any number of arrays

    Returns
    -------
    array : ndarray
        block diagonal array containing `arrays`, the type of which
        is inherited from the first member of `arrays`.
    '''

    arrays = [np.asarray(array) for array in arrays]

    array = arrays[0]

    for i in range(1, len(arrays)):
        array_next = arrays[i]

        zeros_ur = np.zeros((array.shape[0], array_next.shape[1]),
                            dtype=array.dtype)
        zeros_bl = np.zeros((array_next.shape[0], array.shape[1]),
                            dtype=array.dtype)

        array = np.block([[array, zeros_ur],
                          [zeros_bl, array_next]])

    return array


def diagonal(*args, **kwargs):
    ''' Returns a writeable view of a diagonal.

    Parameters
    ----------
    args : list
        positional arguments for np.diagonal, the first of which
        should be an array (n, n)
    kwargs : dictionary
        keyword arguments for np.diagonal

    Returns
    -------
    diag : ndarray (n,n)
        writeable view of the diagonal
    '''

    diag = np.diagonal(*args, **kwargs)
    diag.setflags(write=True)

    return diag


def spin_block(array_a, array_b):
    ''' Returns a 1d, 2d or 4d tensor with alpha and beta blocks.

    Parameters
    ----------
    array_a : (n,n) array
        alpha-spin array
    array_b : (n,n) array
        beta-spin array

    Returns
    -------
    array : (n,n) ndarray
        spin-blocked array where each dimension is double that of the
        input arrays
    '''

    array_a = np.asarray(array_a)
    array_b = np.asarray(array_b)

    if array_a.shape != array_b.shape:
        raise ValueError('Input arrays must have the same shape.')

    if array_a.ndim == 1:
        array = np.concatenate((array_a, array_b), axis=0)
    elif array_a.ndim == 2:
        array = block_diag([array_a, array_b])
    elif array_a.ndim == 4:
        array = np.kron(np.eye(2), np.kron(np.eye(2), array_a).T)
        array[tuple([slice(x, None) for x in array_a.shape])] = array_b
    else:
        raise ValueError('Input arrays must have ndim <= 4.')

    return array


def lanczos(array, v0=None, niter=100): # pragma: no cover
    ''' Lanczos tridiagonalisation of a real symmetric input matrix.

    Parameters
    ----------
    array : (n,n) array
        real symmetric input array
    v0 : (n) array, optional
        Lanczos vector initial guess (default [1 0 ... 0]
    niter : int, optinal
        maximum number of iterations (default 100)

    Returns
    -------
    t : (n,n) ndarray
        tridiagonal form of `array`
    v : (n,niter) ndarray
        Lanczos vectors which project `array` into `t`
    '''

    raise NotImplementedError #FIXME

    array = np.asarray(array, dtype=types.float64)
    n = array.shape[0]

    if v0 is None: 
        v0 = np.eye(n, 1).squeeze()

    v0 = np.asarray(v0, dtype=types.float64)
    v0 /= norm(v0)

    if array.shape != (n, n) or v0.shape != (n,):
        raise ValueError('Input matrix and vector must have equal side '
                         'lengths.')

    if niter > n:
        niter = n

    a = array
    v = np.zeros((n, niter), dtype=types.float64)
    t = np.zeros((n, n), dtype=types.float64)
    vj = np.zeros((n), dtype=types.float64)

    alpha = 0.0
    beta = 0.0

    for j in range(niter-1):
        w = np.dot(a, v0)
        alpha = np.dot(w, v0)

        w -= alpha * v0
        w -= beta * vj

        beta = norm(w)

        vj = v0
        v0 = w / beta

        t[j,j] = alpha
        t[j,j+1] = beta
        t[j+1,j] = beta
        v[:,j] = v0

    w = np.dot(a, v0)
    alpha = np.dot(w, v0)

    w -= alpha * v0
    w -= beta * vj

    t[niter-1,niter-1] = np.dot(w, v0)
    v[:,niter-1] = w / norm(w)

    return t, v


davidson = linalg_helper.davidson1
    

def is_hermitian(array):
    ''' Checks whether an array is symmetric (`array` is real) or 
        Hermitian (`array` is complex).

    Parameters
    ----------
    array : (n,n) array
        input array

    Returns
    -------
    herm : bool
        array is Hermitian
    '''

    herm = np.allclose(array, array.T.conj())

    return herm


def apply_outer(vectors, operator):
    ''' Applies an operation (function) to a set of vectors along new
        axes. Output shape is according to the sizes of the vectors in
        order.

    Parameters
    ----------
    vectors : (n,m) array
        list of vectors to apply operation to
    operator : callable
        function to apply pairwise

    Returns
    -------
    array : (m,) * n ndarray
        array with operation applied pairwise
    '''

    assert all([x.ndim == 1 for x in vectors])

    array = np.copy(vectors[0])

    for i in range(1, len(vectors)):
        v_idx = ((None,) * array.ndim) + (slice(None),)
        array = array[...,None]
        array = operator(array, vectors[i][v_idx])

    return array


def outer_sum(vectors):
    ''' Performs a sum of a set of vectors, each time along a new
        axis.

    Parameters
    ----------
    vectors : (n,m) array
        any number of vectors

    Returns
    -------
    array : (m,) * n ndarray
        array containing outer-summed vectors
    '''

    return apply_outer(vectors, np.add)


def outer_mul(vectors):
    ''' Performs a product of a set of vectors, each time along a new
        axis.

    Parameters
    ----------
    vectors : (n,m) array
        any number of vectors

    Returns
    -------
    array : (m,) * n ndarray
        array containing outer-producted vectors
    '''

    return apply_outer(vectors, np.multiply)


def normalise(array, shift=1e-20):
    ''' Normalises an array column-wise.

    Parameters
    ----------
    array : (n,m) array
        array consisting of column vectors
    shift : float
        small number to prevent zero-vectors flagging a divide-by-zero
        error

    Returns
    -------
    norm_array : (n,m) ndarray
        normalised array
    '''

    array = np.asarray(array)

    n = norm(array, axis=0, keepdims=True)
    n[np.absolute(n) == 0] = shift

    return array / n


def batch_eigh(arrays):
    ''' Diagonalises a batch of matrices and returns a list of eigenvalues
        and eigenvectors. Groups the matrices into those which have the
        same shape in order to exploit vectorisation.

        If the matrices don't have a few shapes in common, this is
        probably quite a bit slower than a simple list comprehension.

    Parameters
    ----------
    arrays : (n,m,m) array
        list of matrices which are to be diagonalised

    Results
    -------
    ws : (n,m) ndarray
        eigenvalues
    vs : (n,m,m) ndarray
        eigenvectors
    '''

    arrays = [np.asarray(array) for array in arrays]
    arrays = np.asarray(arrays)

    num = len(arrays)
    shapes = np.array([x.shape[0] for x in arrays])
    ushapes = np.unique(shapes)

    if len(ushapes) == 1:
        w, v = eigh(np.stack(arrays, axis=0))
        return list(w), list(v)

    mask = np.argsort(shapes)
    inds = np.empty((num), dtype=types.uint32)
    inds[mask] = np.arange(num)

    msort = arrays[mask]
    mshapes = shapes[mask]

    ws = []
    vs = []

    for u in ushapes:
        idx = mshapes == u

        w, v = eigh(np.stack(list(msort[idx]), axis=0))

        ws += list(w)
        vs += list(v)

    ws = np.array(ws)[inds]
    vs = np.array(vs)[inds]

    return ws, vs


def dirsum(key, *arrays): # pragma: no cover
    return pyscf_dirsum(key, *arrays)

    raise NotImplementedError #FIXME fails i,a,j,b->iajb

    ''' Performs a direct sum over a set of arrays in an einsum 
        fashion.

        i.e. dirsum('ij,jk->ijk', a, b) returns an array with
             array[i,j,k] = a[i,j] + b[j,k]

    Parameters
    ----------
    key : str
        einsum-like key, see einsum documentation

    Returns
    -------
    out : ndarray
        resulting array
    '''


    arrays = [np.asarray(array) for array in arrays]

    kin, kout = key.split('->')
    kin = [list(x) for x in kin.split(',')]
    kin_all = sum(kin, [])
    kout = list(kout)
    shapes = [x.shape for x in arrays]
    shapes_all = sum(shapes, ())

    if len(arrays) != len(kin):
        raise ValueError('Number of input keys and arrays must be equal.')

    dims_disagree = [x.ndim != len(k) for x,k in zip(arrays, kin)]
    if any(dims_disagree):
        raise ValueError('Number of dimensions in key and array do not agree '
                         'for argument %d.' % (dims_agree.index(True)))
    
    if set(kin_all) != set(kout):
        raise ValueError('Input keys %s not the same as output keys %s.' %
                         (repr(kin_all), repr(kout)))

    repeated_keys = [len(set(x)) != len(x) for x in kin+kout]
    if any(repeated_keys):
        raise ValueError('Repeated keys not supported (argument %d).' %
                         repeated_keys.index(True))

    for kin_ in set(kin_all):
        shapes_ = [s for s,k in zip(shapes_all, kin_all) if k == kin_]
        if any([s != shapes_[0] for s in shapes_]):
            raise ValueError('Array shapes do not agree for key %s.' % kin_)

    output_shape = [shapes_all[kin_all.index(k)] for k in kout]
    output_type = np.result_type(*[x.dtype for x in arrays])
    out = np.zeros(output_shape, dtype=output_type)

    aux_kin = [tuple([kout.index(x) for x in k]) for k in kin]
    arrays = [x.transpose(np.argsort(y)) for x,y in zip(arrays, aux_kin)]

    for i in range(len(arrays)):
        for j in range(out.ndim):
            if j == arrays[i].ndim or arrays[i].shape[j] != out.shape[j]:
                arrays[i] = np.expand_dims(arrays[i], j)

        out += arrays[i]

    return out


def bypass_empty_ndarray(array, func):
    ''' If `array` is empty, return `np.nan`, else perform the function
        `func` on `array` and return result.

        This prevents numpy killing the program with the error:
            ValueError : zero-size array to reduction operation ...
                         which has no identity
        when i.e. finding a HOMO/LUMO which doesn't exist.

    Parameters
    ----------
    array : array
        array to apply `func` on
    func : callable
        function to apply to `array`

    Returns
    -------
    out : ndarray
        result of `func(array)` if array is not empty, other `np.nan`
    '''

    array = np.asarray(array)

    if array.size == 0:
        return np.nan
    else:
        return func(array)

def amax(x): return bypass_empty_ndarray(x, np.max)
def amin(x): return bypass_empty_ndarray(x, np.min)


def sparsity(array, tol=1e-14):
    ''' Returns the sparsity value of a matrix, higher value means
        more zero elements.

    Parameters
    ----------
    array : array
        input array
    tol : float, optional
        elements with an absolute value below this are assumed zero

    Returns
    -------
    val : float
        sparsity value
    '''

    mask = np.absolute(array) < tol

    return np.sum(mask) / array.size

def density(array, tol=1e-14):
    ''' Returns the density value of a matrix, higher value means
        more zero elements. Equal to 1 - sparsity.
    '''

    return 1.0 - sparsity(array, tol=tol)


def is_iter(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False

def iter_depth(x):
    n = 0
    y = x

    while True:
        if isinstance(y, str):
            break
        elif is_iter(y):
            n += 1
            y = y[0]
        else:
            break

    return n
