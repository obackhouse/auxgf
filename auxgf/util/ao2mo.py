''' Basis transformation routines.
'''

import numpy as np
from pyscf.ao2mo import restore
from pyscf import lib

from auxgf.util import einsum, types


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

    Paramters
    ---------
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

    trans = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', array, c_a, c_b, c_c, c_d)

    return trans


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
    if ndim == (3, 3, 3):
        s, ci, cj = args

        assert s.shape[0] == ci.shape[0] == cj.shape[0]
        nspin = s.shape[0]

        m = [ao2mo_2d(s[i], ci[i], cj[i]) for i in range(nspin)]
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

        m = [[ao2mo_4d(s, ci[i], cj[i], ck[j], cl[j]) 
              for j in range(nb)] for i in range(na)]
        return np.stack(m)

    # spin 4d tensor with spin coefficients: (6, 3, 3, 3, 3)
    if ndim == (6, 3, 3, 3, 3):
        s, ci, cj, ck, cl = args

        assert s.shape[0] == ci.shape[0] == cj.shape[0]
        assert s.shape[1] == ck.shape[0] == cl.shape[0]
        na = s.shape[0]
        nb = s.shape[1]

        m = [[ao2mo_4d(s[i,j], ci[i], cj[i], ck[j], cl[j]) 
              for j in range(nb)] for i in range(na)]

        return np.stack(m)

    raise ValueError('Inputs with dimensions %s not supported for generalized'
                     ' ao2mo function.' % repr(ndim))


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

    trans = lib.einsum('pqrs,qi,rj,sk->pijk', array, c_a, c_b, c_c)

    return trans

mo2qo = mo2qo_4d


class SemiDirectMO2QO:
    ''' Class for a semi-direct transformation of a 4-dimensional
        tensor from MO to QO basis. 

        Object stores (xp|jk), (xi|rk) and (xi|js). The array which
        provides the most memory-efficient route to the target slice
        will be used in the transformation.

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
    build : bool, optional
        if False, do not build the object (default True)

    Attributes
    ----------
    pijk : (p,i,j,k) ndarray
        transformed array
    nphys : int
        number of physical degrees of freedom (`p`).
    c : list of ndarray
        list of vectors to transform `array`
    eri : list of ndarray
        list of intermediate-transformed arrays

    Methods
    -------
    __getitem__(tup)
        provides a non-direct tensor resulting from array indexing
        with `tup`, which contains any value which is a valid for
        indexing an ndarray.
    getitem_i(i)
        provides a non-direct tensor resulting from array indexing
        with `[:,i,:,:]`, where `i` is an int.
    getitem_j(j)
        provides a non-direct tensor resulting from array indexing
        with `[:,:,j,:]`, where `j` is an int.
    getitem_k(k)
        provides a non-direct tensor resulting from array indexing
        with `[:,:,:,k]`, where `k` is an int. 
    get_slice(s1, s2, s3)
        provides a new SemiDirectMO2QO object containing result of
        applying index slices `s1`, `s2` and `s3` to the `pijk`.
    '''

    def __init__(self, array, c_a, c_b, c_c, build=True):
        if build:
            self.ci = c_a
            self.cj = c_b
            self.ck = c_c

            self.xpjk = einsum('xpqr,qj,rk->xpjk', array, c_b, c_c)
            self.xiqk = einsum('xpqr,pi,rk->xiqk', array, c_a, c_c)
            self.xijr = einsum('xpqr,pi,qj->xijr', array, c_a, c_b)

            self.shape = (self.ci.shape[0], self.ci.shape[1], 
                          self.cj.shape[1], self.ck.shape[1])

    def __getitem__(self, tup):
        idx = []

        for x in tup:
            if isinstance(x, (list, np.ndarray, slice)):
                idx.append(x)
            elif isinstance(x, tuple):
                idx.append(list(x))
            else:
                idx.append([x,])

        while len(idx) < 4:
            idx.append(slice(None))

        # Definitely a less expensive way to analyse the cost of
        # these masks #TODO:
        cost = []
        for i in range(3):
            mask = np.ones((self.c[i].shape[1]), dtype=types.int16)
            cost.append(mask[idx[i+1]].shape[0])
        cmin = np.argmin(cost)

        x, i, j, k = idx

        if cmin == 0:
            xpjk = self.xpjk[:,:,:,k][:,:,j][x]
            xijk = einsum('xpjk,pi->xijk', xpjk, self.ci[:,i])
        elif cmin == 1:
            xiqk = self.xiqk[:,:,:,k][:,i][x]
            xijk = einsum('xiqk,qj->xijk', xiqk, self.cj[:,j])
        elif cmin == 2:
            xijr = self.xijr[:,:,j][:,i][x]
            xijk = einsum('xijr,rk->xijk', xijr, self.ck[:,k])

        return np.squeeze(xijk)

    def getitem_i(self, i):
        ''' Optimized equivalent of self[:,i,:,:], where i is an
            integer.
        '''

        return einsum('xpjk,p->xjk', self.xpjk, self.ci[:,i])

    def getitem_j(self, j):
        ''' Optimized equivalent of self[:,:,j,:] where j is an
            integer.
        '''

        return einsum('xiqk,q->xik', self.xiqk, self.cj[:,j])

    def getitem_ij(self, i, j):
        ''' Optimized equivalent of self[:,i,j,:] where i,j are
            integers.
        '''
        
        return einsum('xqk,q->xk', self.xiqk[:,i], self.cj[:,j])

    def getitem_k(self, k):
        ''' Optimized equivalent of self[:,:,:,k] where k is an
            integer.
        '''

        return einsum('xijr,r->xij', self.xijr, self.ck[:,k])

    #def __getitem__(self, tup):
    #    ''' This function removes the semi-directness of the tensor.
    #        To return a semi-direct slice, use get_slice instead.
    #    '''

    #    idx = []

    #    for x in tup:
    #        if isinstance(x, (list, np.ndarray, slice)):
    #            idx.append(x)
    #        elif isinstance(x, tuple):
    #            idx.append(list(x))
    #        else:
    #            idx.append([x,])

    #    while len(idx) < 4:
    #        idx.append(slice(None))

    #    a, b, c, d = idx
    #    
    #    pqjk = self.pqjk
    #    pqjk = pqjk[:,:,:,d][:,:,c][a]
    #    pijk = einsum('pqjk,qi->pijk', pqjk, self.ci[:,b])

    #    return np.squeeze(pijk)

    #def get_slice(self, s1, s2, s3):
    #    ci = self.ci[:,s1]
    #    pqjk = self.pqjk[:,:,:,s3][:,:,s2]

    #    eri = SemiDirectMO2QO(None, None, None, None, build=False)
    #    eri.ci = ci
    #    eri.pqjk = pqjk

    #    return eri

    def get_slice(self, s1, s2, s3):
        eri = SemiDirectMO2QO(None, None, None, None, build=False)

        ci = self.ci[:,s1]
        xpjk = self.xpjk[:,:,:,s3][:,:,s2]
        eri.ci = ci
        eri.xpjk = xpjk

        cj = self.cj[:,s2]
        xiqk = self.xiqk[:,:,:,s3][:,s1]
        eri.cj = cj
        eri.xiqk = xiqk

        ck = self.ck[:,s3]
        xijr = self.xijr[:,:,s2][:,s1]
        eri.ck = ck
        eri.xijr = xijr

        eri.shape = (eri.ci.shape[0], eri.ci.shape[1],
                     eri.cj.shape[1], eri.ck.shape[1])

        return eri

    @property
    def nphys(self):
        return self.ci.shape[0]

    @property
    def c(self):
        return [self.ci, self.cj, self.ck]

    @property
    def eri(self):
        return [self.xpjk, self.xiqk, self.xijr]

    def copy(self):
        eri = SemiDirectMO2QO(None, None, None, None, build=False)

        eri.ci = ci.copy() 
        eri.cj = cj.copy()
        eri.ck = ck.copy()

        eri.xpjk = xpjk.copy() 
        eri.xiqk = xiqk.copy()
        eri.xijr = xijr.copy()

        eri.shape = (eri.ci.shape[0], eri.ci.shape[1],
                     eri.cj.shape[1], eri.ck.shape[1])

        return eri















