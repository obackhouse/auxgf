''' Attempts to load the Intel MKL library and wrap some of the
    LAPACKE functions as python functions.

    Arrays are passed as row-major (C-contiguous).
'''

import numpy as np
import os
from ctypes import *


_ndp = np.ctypeslib.ndpointer
_lapacke_row_major = 101
_lapacke_col_major = 102


lib_mkl = None
has_mkl = False

try:
    # Try to load library for the $MKLROOT environment variable

    mklroot = os.environ['MKLROOT']
    path = mklroot + '/lib/intel64/libmkl_rt.so'

    lib_mkl = np.ctypeslib.load_library('libmkl_rt.so', path)
    has_mkl = True

except:
    try:
        # If the previous failed, try to search for MKL in the 
        # $LD_LIBRARY_PATH environment variable

        ld = os.environ['LD_LIBRARY_PATH']
        ld = ld.split(':')

        for path in ld:
            if 'mkl/lib/intel64' in path:
                path += '/libmkl_rt.so'
                lib_mkl = np.ctypeslib.load_library('libmkl_rt.so', path)
                has_mkl = True
                break

    except:
        pass


if has_mkl:
    # DGEQRF
    lapacke_dgeqrf = lib_mkl.LAPACKE_dgeqrf
    lapacke_dgeqrf.argtypes = [
        c_int32,
        c_int64,
        c_int64,
        _ndp(ndim=2, flags='C,A,W', dtype=np.float64),
        c_int64,
        _ndp(ndim=1, flags='C,A,W', dtype=np.float64) 
    ]
    lapacke_dgeqrf.restype = c_int64

    # DORGQR
    lapacke_dorgqr = lib_mkl.LAPACKE_dorgqr
    lapacke_dorgqr.argtypes = [
        c_int32,
        c_int64,
        c_int64,
        c_int64,
        _ndp(ndim=2, flags='C,A,W', dtype=np.float64),
        c_int64,
        _ndp(ndim=1, flags='C,A,W', dtype=np.float64)
    ]
    lapacke_dorgqr.restype = c_int64


def _wrap_lapacke(method):
    ''' Decorator for wrapped functions which asserts that the
        library exists before calling the function, and that the
        function exited successfully.

        Actual functions don't bother returning `info`, even if
        the definitions imply so.
    '''

    global has_mkl, lib_mkl

    def wrapper(*args):
        if not has_mkl or lib_mkl is None:
            raise RuntimeError('MKL library must be present to use '
                               'auxgf.util.mkl.')

        info = method(*args)

        if info != 0:
            raise RuntimeError('MKL function did not return info == 0.')

    return wrapper


def _set_mkl_num_threads(method):
    ''' Temporarily sets MKL_NUM_THREADS to be OMP_NUM_THREADS. The
        MKL functions seem to usually use the former, so this seems
        like a decent way to ensure the MKL functions use as many
        threads as the standard numpy ones.
    '''

    #TODO come up with a more rigorous solution to this

    mkl_num_threads = os.environ.get('MKL_NUM_THREADS', 1)
    omp_num_threads = os.environ.get('OMP_NUM_THREADS', 1)

    def wrapper(*args):
        if mkl_num_threads == omp_num_threads:
            out = method(*args)
        else:
            try:
                os.environ['MKL_NUM_THREADS'] = omp_num_threads
                out = method(*args)
            finally:
                os.environ['MKL_NUM_THREADS'] = mkl_num_threads

    return wrapper


@_set_mkl_num_threads
@_wrap_lapacke
def dgeqrf(m, n, a, lda, tau):
    return lapacke_dgeqrf(c_int32(_lapacke_row_major),
                          c_int64(m),
                          c_int64(n),
                          a,
                          c_int64(lda),
                          tau)


@_set_mkl_num_threads
@_wrap_lapacke
def dorgqr(m, n, k, a, lda, tau):
    return lapacke_dorgqr(c_int32(_lapacke_row_major),
                          c_int64(m),
                          c_int64(n),
                          c_int64(k),
                          a,
                          c_int64(lda),
                          tau)







