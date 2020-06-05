# some mpi parallel adc(2) shit bro
# algo/eqns mix of my RAGF2 stuff and work due to Banerjee & Sokolv: https://arxiv.org/abs/1910.07116

from auxgf import *
import numpy as np
from mpi4py import MPI
from pyscf import lib
from pyscf import adc as pyscf_adc
from auxgf.future.agf2.mpi_agf2_1_1 import df_ao2mo
from scipy.sparse.linalg import eigsh, LinearOperator
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

reshape_internal = lambda x, s1, swap, s2 : x.reshape(s1).swapaxes(*swap).reshape(s2)
is_contiguous = util.linalg._is_contiguous


# don't trust einsum 

def dgemm(a, b, c=None, alpha=1, beta=0):
    if not (is_contiguous(a) and is_contiguous(b)):
        m = alpha * np.dot(a, b)
        if out is not None:
            m += beta * out
        return m

    m = util.dgemm(a, b, c=c, alpha=alpha, beta=beta)

    return m

def iab_jab__ij(iab, jab, out=None, alpha=1, beta=0):
    ni, na, nb = iab.shape
    assert jab.shape == (ni, na, nb)

    iab = iab.reshape((ni, na*nb))
    jab = jab.reshape((ni, na*nb))

    return dgemm(iab, jab.T, c=out, alpha=alpha, beta=beta)

def iab_jba__ij(iab, jba, out=None, alpha=1, beta=0):
    ni, na, nb = iab.shape
    assert jba.shape == (ni, nb, na)

    abj = jba.transpose(2,1,0)

    iab = iab.reshape((ni, na*nb))
    abj = abj.reshape((na*nb, ni))

    return dgemm(iab, abj, c=out, alpha=alpha, beta=beta)

def aib_jab__ij(aib, jab, out=None, alpha=1, beta=0):
    # Requires copy atm :(

    na, ni, nb = aib.shape
    assert jab.shape == (ni, na, nb)

    iab = aib.swapaxes(0,1)

    iab = iab.reshape((ni, na*nb))
    jab = jab.reshape((ni, na*nb))

    return dgemm(iab, jab.T, c=out, alpha=alpha, beta=beta)


def get_1p(eo, ev, qia, chempot=0.0):
    # Get the 1p space of the Hamiltonian

    nocc, nvir = eo.size, ev.size
    e_ia = eo[:,None] - ev[None,:]

    m = np.zeros((nocc, nocc))

    for k in range(rank, nocc, size):
        iab = np.dot(qia.T, qia[:,k*nvir:(k+1)*nvir])
        aib = iab.T

        iab = iab.reshape((nocc, nvir, nvir))
        aib = aib.reshape((nvir, nocc, nvir))

        t2 = iab.copy()
        t2 /= e_ia[:,:,None] + e_ia[k][None,None,:]
        t2a = t2 - t2.swapaxes(1,2).copy()

        m = iab_jab__ij(ev[None,:,None] * t2a, t2a, out=m, beta=1)
        m = iab_jab__ij(ev[None,:,None] * t2, t2, out=m, beta=1)
        m = iab_jab__ij(ev[None,None,:] * t2, t2, out=m, beta=1)

        m = iab_jab__ij(eo[:,None,None] * t2a, t2a, out=m, alpha=-0.25, beta=1)
        m = iab_jab__ij(eo[:,None,None] * t2, t2, out=m, alpha=-0.5, beta=1)

        m = iab_jab__ij(t2a, eo[:,None,None] * t2a, out=m, alpha=-0.25, beta=1)
        m = iab_jab__ij(t2, eo[:,None,None] * t2, out=m, alpha=-0.5, beta=1)

        m = iab_jab__ij(eo[k] * t2a, t2a, out=m, alpha=-0.5, beta=1)
        m = iab_jab__ij(eo[k] * t2, t2, out=m, alpha=-1.0, beta=1)

        m = iab_jab__ij(t2a, iab, out=m, alpha=0.5, beta=1)
        m = iab_jba__ij(t2a, iab, out=m, alpha=-0.5, beta=1)
        m = iab_jab__ij(t2, iab, out=m, beta=1)

        m = iab_jab__ij(iab, t2a, out=m, alpha=0.5, beta=1)
        m = aib_jab__ij(aib, t2a, out=m, alpha=-0.5, beta=1)
        m = iab_jab__ij(iab, t2, out=m, beta=1)

    m = util.mpi.reduce(m)
    m += np.diag(eo)

    return m


def get_dot(eo, ev, ijq, qia, mij, chempot=0.0):
    # Get the dot product function between the Hamiltonian and a vector

    nocc = eo.size
    nvir = ev.size

    indices = util.mpi.tril_indices_rows(nocc)
    e_ia = eo[:,None] - ev[None,:]

    s_i = slice(None, nocc)
    s_ia = slice(nocc, None)

    pos_factor = np.sqrt(0.5)
    neg_factor = np.sqrt(1.5)

    size = nocc + nocc*nocc*nvir

    def matvec(vec):
        vec = np.asarray(vec)
        shape_in = vec.shape
        vec = vec.reshape((nocc + nocc*nocc*nvir, -1))
        out = np.zeros_like(vec)

        for j in indices[rank]:
            p1 = nvir * j * j

            iq = ijq[j*nocc:(j+1)*nocc]
            qa = qia[:,j*nvir:(j+1)*nvir]

            vjka = np.dot(iq, qia[:,:j*nvir]).reshape((nocc, -1))
            vkja = np.dot(ijq[:j*nocc], qa)
            vkja = reshape_internal(vkja, (j, nocc, nvir), (0,1), (nocc, j*nvir))
            vkja = vkja.reshape((nocc, -1))
            vjja = np.dot(iq, qa)

            ea = eb = eo[j] + e_ia[:j].ravel()
            ec = 2 * eo[j] - ev

            va = neg_factor * (vjka - vkja)
            vb = pos_factor * (vjka + vkja)
            vc = vjja

            for e,v in zip([ea, eb, ec], [va, vb, vc]):
                p0, p1 = p1, p1 + e.size
                s_ia_blk = slice(nocc+p0, nocc+p1)

                out[s_i] += np.dot(v, vec[s_ia_blk])
                out[s_ia_blk] += np.dot(vec[s_i].T, v).T
                out[s_ia_blk] += (e[:,None] - chempot) * vec[s_ia_blk]

        out = util.mpi.reduce(out)
        out[s_i] += np.dot(mij, vec[s_i])
        out = out.reshape(shape_in)

        return out

    return matvec, size


def get_diag(eo, ev, mij, chempot=0.0):
    # Get the diagonal of the Hamiltonian

    nocc = eo.size
    nvir = ev.size

    indices = util.mpi.tril_indices_rows(nocc)

    s_i = slice(None, nocc)
    s_ia = slice(nocc, None)

    diag = np.zeros((nocc+nocc*nocc*nvir))

    for j in indices[rank]:
        p1 = nvir * j * j

        ea = eb = eo[j] + np.subtract.outer(eo[:j], ev).ravel()
        ec = 2 * eo[j] - ev

        for e in [ea, eb, ec]:
            p0, p1 = p1, p1 + e.size
            s_ia_blk = slice(nocc+p0, nocc+p1)

            diag[s_ia_blk] = e

    diag = util.mpi.reduce(diag)
    diag[s_i] = np.diag(mij)

    return diag


def get_guess(diag, nroots=1):
    ndim = diag.size
    guess = np.zeros((nroots, ndim))
    mask = np.argsort(np.absolute(diag))

    for i in range(nroots):
        guess[i,mask[i]] = 1

    return list(guess)


def pick(w, v, nroots, callback):
    if nroots > len(w):
        mask = np.argpartition(np.absolute(w), nroots)
        w = w[mask]
        v = v[:,mask]

    mask = np.argsort(np.absolute(w))
    w = w[mask]
    v = v[:,mask]

    return w, v, 0


def diagonalise(matvec, diag, size, method='pyscf', nroots=1, tol=1e-14, ntrial=None, maxiter=None):
    if maxiter is None: maxiter = 10 * size
    if ntrial is None: ntrial = min(size, max(2*nroots+1, 20))

    #TODO: find a better parallel sparse iterative diagonaliser... slepc? petsc?

    if method == 'scipy':
        linop = LinearOperator(shape=(size, size), dtype=np.float64, matvec=matvec)
        w, v = eigsh(linop, k=nroots, which='SM', tol=tol, ncv=ntrial, maxiter=maxiter)

    elif method == 'pyscf':
        guess = get_guess(diag, nroots=nroots)
        conv, w, v = lib.davidson1(matvec, guess, diag, tol=tol, nroots=nroots, max_space=ntrial, max_cycle=maxiter, pick=pick)

        if not np.all(conv):
            util.log.warn('PySCF Davidson solver did not converge.')

    return w, v


def run(rhf, excite='ip', method='pyscf', nroots=1, tol=1e-14, debug=False):
    if excite == 'ip':
        o, v = rhf.occ > 0, rhf.occ == 0
        factor = -1
    elif excite == 'ea':
        v, o = rhf.occ > 0, rhf.occ == 0
        factor = 1

    nocc, nvir = np.sum(o), np.sum(v)
    eo, ev = rhf.e[o], rhf.e[v]
    co, cv = rhf.c[:,o], rhf.c[:,v]

    eri = rhf._pyscf.with_df._cderi
    qia = df_ao2mo(eri, co, cv, sym_in='s2', sym_out='s1')
    ijq = df_ao2mo(eri, co, co, sym_in='s2', sym_out='s2').T
    ijq = np.ascontiguousarray(lib.unpack_tril(ijq, axis=0).reshape((nocc*nocc, -1)))

    mij = get_1p(eo, ev, qia)
    diag = get_diag(eo, ev, mij)
    matvec, size = get_dot(eo, ev, ijq, qia, mij)

    if debug:
        return matvec, diag, mij

    w, v = diagonalise(matvec, diag, size, method=method, nroots=nroots, tol=tol)
    w *= factor

    return w, v


if __name__ == '__main__':
    if 0:
        rhf = hf.RHF(mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='aug-cc-pvdz'), with_df=True).run()
        import IPython
        ipython = IPython.get_ipython()
        ipython.magic('load_ext line_profiler')
        matvec = run(rhf, debug=True)[0]
        ipython.magic('lprun -f matvec run(rhf, method="pyscf")')

    if 1:
        rhf = hf.RHF(mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='aug-cc-pvdz'), with_df=True, auxbasis='aug-cc-pv5z').run()

        t0 = time.time()

        ip = run(rhf, nroots=1, method='pyscf')[0][0]
        if rank == 0:
            print(ip)

        t1 = time.time()
        if rank == 0:
            print(t1-t0, '\n')

        rhf = hf.RHF(rhf.mol, with_df=False).run()

        t0 = time.time()

        adc2 = adc.RADC2(rhf, nroots=1, wtol=0, verbose=False).run()
        if rank == 0:
            print(adc2.ip[0][0])

        t1 = time.time()
        if rank == 0:
            print(t1-t0, '\n')

        t0 = time.time()

        adc2 = pyscf_adc.ADC(rhf._pyscf).run()
        if rank == 0:
            print(pyscf_adc.uadc.UADCIP(adc2).kernel()[0])

        t1 = time.time()
        if rank == 0:
            print(t1-t0, '\n')


