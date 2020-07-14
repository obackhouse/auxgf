''' Basis transformation routines.
'''

import numpy as np
from pyscf import lib as _pyscf_lib
from pyscf import ao2mo as _pyscf_ao2mo
import ctypes
import functools

from auxgf.util import einsum, types


restore = _pyscf_ao2mo.restore
conc_mos = _pyscf_ao2mo.incore._conc_mos


def ao2mo_2d(array, c_a, c_b): 
    ''' Transforms the basis of a 2d array.

    Parameters
    ----------
    array : (p,q) array
        array to be transformed
    c_a : (p,i) array
        vectors to transform first dimension of `array`
    c_b : (q,j) array
        vectors to transform second dimension of `array`

    Returns
    -------
    trans : (i,j) ndarray
        transformed array
    '''

    trans = einsum('pq,pi,qj->ij', array, c_a, c_b)

    return trans


def ao2mo_4d(array, c_a, c_b, c_c, c_d):
    ''' Transforms the basis of a 4d array. Overrides use_pyscf_einsum
        flag in auxgf.util.linalg because pyscf.lib.einsum is quicker
        here.

    Parameters
    ----------
    array : (p,q,r,s) array
        array to be transformed
    c_a : (p,i) array
        vectors to transform first dimension of `array`
    c_b : (q,j) array
        vectors to transform second dimension of `array`
    c_c : (r,k) array
        vectors to transform third dimension of `array`
    c_d : (s,l) array
        vectors to transform fourth dimension of `array`

    Returns
    -------
    trans : (i,j,k,l) ndarray
        transformed array
    '''

    trans = _pyscf_lib.einsum('pqrs,pi,qj,rk,sl->ijkl', array, c_a, c_b, c_c, c_d)

    return trans


_fdrv = functools.partial(_pyscf_ao2mo._ao2mo.libao2mo.AO2MOnr_e2_drv, 
                          _pyscf_ao2mo._ao2mo.libao2mo.AO2MOtranse2_nr_s2,
                          _pyscf_ao2mo._ao2mo.libao2mo.AO2MOmmm_bra_nr_s2)

_nr_e2 = _pyscf_ao2mo._ao2mo.nr_e2

to_ptr = lambda m : m.ctypes.data_as(ctypes.c_void_p)

def ao2mo_df(array, c_a, c_b, out=None):
    ''' Transforms the basis of a density fitted Cholesky ERI tensor.

    Parameters
    ----------
    array : (t,p,q) array
        array to be transformed
    c_a : (p,i) array
        vectors to transform the first dimension of `array`
    c_b : (q,j) array
        vectors to transform the second dimension of `array`

    Returns
    -------
    trans : (t,i,j) ndarray
        transformed array
    '''

    naux = array.shape[0]
    ijsym, nij, cij, sij = conc_mos(c_a, c_b, compact=True)
    i, j = c_a.shape[1], c_b.shape[1]

    if out is None:
        out = np.zeros((naux, i*j), dtype=types.float64)

    array = array.reshape((naux, -1))

    out = _nr_e2(array, cij, sij, out=out, aosym='s1', mosym='s1')

    out = out.reshape((naux, i, j))

    return out


def ao2mo(*args):
    ''' AO to MO basis transformation wrapper. Very generalized and 
        uses the shape of the input arrays to determine the desired
        transformation. This can probably be simplified.

    Parameters
    ----------
    array : array
        array to be transformed
    c : array
        vectors to transform array

    Returns
    -------
    trans : ndarray
        transformed array
    '''

    ndim = tuple([x.ndim for x in args])

    # spin-free 2d matrix with spin-free coefficients: (2, 2, 2)
    if ndim == (2, 2, 2):
        s, ci, cj = args
        return ao2mo_2d(s, ci, cj)

    # spin-free 2d matrix with spin coefficients: (2, 3, 3)
    if ndim == (2, 3, 3):
        s, ci, cj = args

        assert ci.shape[0] == cj.shape[0]
        nspin = ci.shape[0]

        m = [ao2mo_2d(s, ci[i], cj[i]) for i in range(nspin)]
        return np.stack(m)

    # spin 2d matrix with spin coefficients: (3, 3, 3)
    if ndim == (3, 3, 3) and args[0].shape[0] == 2:
        s, ci, cj = args

        assert s.shape[0] == ci.shape[0] == cj.shape[0]
        nspin = s.shape[0]

        m = [ao2mo_2d(s[i], ci[i], cj[i]) for i in range(nspin)]
        return np.stack(m)

    # spin-free 3d tensor with spin-free coefficients: (3, 2, 2)
    if ndim == (3, 2, 2):
        s, ci, cj = args
        return ao2mo_df(s, ci, cj)

    # spin-free 3d tensor with spin coefficients: (3, 3, 3)
    if ndim == (3, 3, 3):
        s, ci, cj = args

        assert ci.shape[0] == cj.shape[0]
        nspin = ci.shape[0]

        m = [ao2mo_df(s, ci[i], cj[i]) for i in range(nspin)]
        return np.stack(m)

    # spin 3d tensor with spin coefficients: (4, 3, 3)
    if ndim == (4, 3, 3):
        s, ci, cj = args

        assert s.shape[0] == ci.shape[0] == cj.shape[0]
        nspin = s.shape[0]

        m = [ao2mo_df(s[i], ci[i], cj[i]) for i in range(nspin)]
        return np.stack(m)

    # spin-free 4d tensor with spin-free coefficients: (4, 2, 2, 2)
    if ndim == (4, 2, 2, 2, 2):
        s, ci, cj, ck, cl = args
        return ao2mo_4d(s, ci, cj, ck, cl)

    # spin-free 4d tensor with spin coefficients: (4, 3, 3, 3, 3)
    if ndim == (4, 3, 3, 3, 3):
        s, ci, cj, ck, cl = args

        assert ci.shape[0] == cj.shape[0]
        assert ck.shape[0] == cl.shape[0]
        na = ci.shape[0]
        nb = ck.shape[0]

        m = [[ao2mo_4d(s, ci[i], cj[i], ck[j], cl[j]) for j in range(nb)] for i in range(na)]
        return np.stack(m)

    # spin 4d tensor with spin coefficients: (6, 3, 3, 3, 3)
    if ndim == (6, 3, 3, 3, 3):
        s, ci, cj, ck, cl = args

        assert s.shape[0] == ci.shape[0] == cj.shape[0]
        assert s.shape[1] == ck.shape[0] == cl.shape[0]
        na = s.shape[0]
        nb = s.shape[1]

        m = [[ao2mo_4d(s[i,j], ci[i], cj[i], ck[j], cl[j]) for j in range(nb)] for i in range(na)]

        return np.stack(m)

    raise ValueError('Inputs with dimensions %s not supported for generalized ao2mo function.' 
                     % repr(ndim))


def mo2qo_4d(array, c_a, c_b, c_c):
    ''' Three-quarter transformation of the basis of a 4d array. 
        Overrides use_pyscf_einsum flag in auxgf.util.linalg because 
        pyscf.lib.einsum is quicker here.

    Parameters
    ----------
    array : (p,q,r,s) array
        array to be transformed
    c_a : (q,i) array
        vectors to transform second dimension of `array`
    c_b : (r,j) array
        vectors to transform third dimension of `array`
    c_c : (s,k) array
        vectors to transform fourth dimension of `array`

    Returns
    -------
    trans : (p,i,j,k) ndarray
        transformed array
    '''

    #return _pyscf_lib.einsum('pqrs,qi,rj,sk->pijk', array, c_a, c_b, c_c)

    p, q, r, s = array.shape
    i = c_a.shape[-1]
    j = c_b.shape[-1]
    a = c_c.shape[-1]

    if a > i:
        # p,q,r,s -> pqs,r
        trans = array.swapaxes(2,3).reshape(p*q*s, r)
        # pqs,r -> pqs,j
        trans = np.dot(trans, c_b)
        # pqs,j -> sjp,q
        trans = trans.reshape(p*q, s*j).T.reshape(s*j*p, q)
        # sjp,q -> sjp,i
        trans = np.dot(trans, c_a)
        # sjp,i -> pij,s
        trans = trans.reshape(s*j, p*i).T.reshape(p*i, s, j).swapaxes(1,2).reshape(p*i*j, s)
        # pij,s -> pij,a
        trans = np.dot(trans, c_c)
        # pij,a -> p,i,j,a
        trans = trans.reshape(p, i, j, a)
    else:
        # p,q,r,s -> pqr,s
        trans = array.reshape(p*q*r, s)
        # pqr,s -> pqr,a
        trans = np.dot(trans, c_c)
        # pqr,a -> pqa,r
        trans = trans.reshape(p*q, r, a).swapaxes(1,2).reshape(p*q*a, r)
        # pqa,r -> pqa,j
        trans = np.dot(trans, c_b)
        # pqa,j -> ajp,q
        trans = trans.reshape(p*q, a*j).T.reshape(a*j*p, q)
        # ajp,q -> ajp,i
        trans = np.dot(trans, c_a)
        # ajp,i -> p,i,j,a
        trans = trans.reshape(a*j, p*i).T.reshape(p, i, a, j).swapaxes(2,3)

    return np.ascontiguousarray(trans)

mo2qo = mo2qo_4d
