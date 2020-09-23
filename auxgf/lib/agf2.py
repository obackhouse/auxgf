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

    except OSError as e:
        clib = None
        print(e)
        raise RuntimeError('Cannot locate %s C library.' % name)

    return clib


_libagf2 = _load('agf2')

if _libagf2 is not None:
    _libagf2.build_part_loop_rhf.argtypes = [
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

    _libagf2.build_part_loop_uhf.argtypes = [
            types.float64.ndpointer(2, 'C,A'),
            types.float64.ndpointer(2, 'C,A'),
            types.float64.ndpointer(2, 'C,A'),
            types.float64.ndpointer(1, 'C,A'),
            types.float64.ndpointer(1, 'C,A'),
            types.float64.ndpointer(1, 'C,A'),
            types.float64.ndpointer(1, 'C,A'),
            types.uint32.ctype,
            types.uint32.ctype,
            types.uint32.ctype,
            types.uint32.ctype,
            types.uint32.ctype,
            types.uint32.ctype,
            types.uint32.ctype,
            types.uint32.ctype,
            types.float64.ndpointer(2, 'C,A,W'),
            types.float64.ndpointer(2, 'C,A,W')]


def build_part_loop_rhf(ixq, qja, gf_occ, gf_vir, istart, iend, vv=None, vev=None):
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

    _libagf2.build_part_loop_rhf(ixq, qja, ei, ea, nphys, nocc, nvir, naux, istart, iend, vv, vev)

    return vv, vev


def build_part_loop_uhf(ixq, qja, gf_occ, gf_vir, istart, iend, vv=None, vev=None):
    ''' Compute the inner loop of the OptUAGF2.build_part function with
        OpenMP parallelism.
    '''

    nocc_a = gf_occ[0].naux
    nocc_b = gf_occ[1].naux
    nvir_a = gf_vir[0].naux
    nvir_b = gf_vir[1].naux
    nphys = gf_occ[0].nphys

    ei_a = gf_occ[0].e
    ei_b = gf_occ[1].e
    ea_a = gf_vir[0].e
    ea_b = gf_vir[1].e

    ixq_a = ixq[0].reshape(nocc_a*nphys, -1)
    qja_a = qja[0].reshape(-1, nocc_a*nvir_a)
    qja_b = qja[1].reshape(-1, nocc_b*nvir_b)

    naux = qja_a.shape[0]

    if vv is None:
        vv = np.zeros((nphys, nphys), dtype=types.float64, order='C')

    if vev is None:
        vev = np.zeros((nphys, nphys), dtype=types.float64, order='C')

    ixq_a = np.ascontiguousarray(ixq_a)
    qja_a = np.ascontiguousarray(qja_a)
    qja_b = np.ascontiguousarray(qja_b)
    nphys = types.uint32.ctype(nphys)
    nocc_a = types.uint32.ctype(nocc_a)
    nocc_b = types.uint32.ctype(nocc_b)
    nvir_a = types.uint32.ctype(nvir_a)
    nvir_b = types.uint32.ctype(nvir_b)
    istart = types.uint32.ctype(istart)
    iend = types.uint32.ctype(iend)
    vv = np.ascontiguousarray(vv)
    vev = np.ascontiguousarray(vev)

    _libagf2.build_part_loop_uhf(ixq_a, qja_a, qja_b, ei_a, ei_b, ea_a, ea_b, nphys, nocc_a, nocc_b, nvir_a, nvir_b, naux, istart, iend, vv, vev)

    return vv, vev
