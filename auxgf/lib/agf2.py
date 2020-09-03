''' Functions to load OptRAGF2 C library
'''

import numpy as np
import ctypes
import os

from auxgf.util import types


def _load(name):
    cwd = os.path.dirname(os.path.realpath(__file__))

    try:
        path = os.path.join(cwd, 'lib%s/%s.so' % (name, name))
        clib = np.ctypeslib.load_library(name, path)

    except OSError:
        clib = None
        #raise RuntimeError('Cannot locate %s C library.' % name)

    return clib


_liboptragf2 = _load('agf2')

if _liboptragf2 is not None:
    _liboptragf2.build_part_loop.argtypes = [
            types.float64.ndpointer(2, 'C,A'),
            types.float64.ndpointer(2, 'C,A'),
            types.float64.ndpointer(1, 'C,A'),
            types.float64.ndpointer(1, 'C,A'),
            types.uint32.ctype,
            types.uint32.ctype,
            types.uint32.ctype,
            types.uint32.ctype,
            types.uint32.ctype,
            types.uint32.ctype,
            types.float64.ndpointer(2, 'C,A,W'),
            types.float64.ndpointer(2, 'C,A,W')]

def build_part_loop(ixq, qja, gf_occ, gf_vir, istart, iend, vv=None, vev=None):
    ''' Compute the inner loop of the OptRAGF2.build_part function with
        OpenMP parallelism.
    '''

    nocc = gf_occ.naux
    nvir = gf_vir.naux
    nphys = gf_occ.nphys

    ei = gf_occ.e
    ea = gf_vir.e

    ixq = ixq.reshape(nocc*nphys, -1)
    qja = qja.reshape(-1, nocc*nvir)

    naux = qja.shape[0]

    if vv is None:
        vv = np.zeros((nphys, nphys), dtype=types.float64, order='C')

    if vev is None:
        vev = np.zeros((nphys, nphys), dtype=types.float64, order='C')

    ixq = np.ascontiguousarray(ixq)
    qja = np.ascontiguousarray(qja)
    nphys = types.uint32.ctype(nphys)
    nocc = types.uint32.ctype(nocc)
    nvir = types.uint32.ctype(nvir)
    naux = types.uint32.ctype(naux)
    istart = types.uint32.ctype(istart)
    iend = types.uint32.ctype(iend)
    vv = np.ascontiguousarray(vv)
    vev = np.ascontiguousarray(vev)

    _liboptragf2.build_part_loop(ixq, qja, ei, ea, nphys, nocc, nvir, naux, istart, iend, vv, vev)

    return vv, vev
