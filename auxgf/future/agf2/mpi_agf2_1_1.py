from auxgf import *
import numpy as np
from mpi4py import MPI
from pyscf import lib, ao2mo, df
from pyscf.scf import jk
from pyscf.df import df_jk
from scipy import optimize
from scipy.linalg import blas
import ctypes

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

reshape_internal = lambda x, s1, swap, s2 : x.reshape(s1).swapaxes(*swap).reshape(s2)

DF_MAXBLK = 120
OPT_MAXITER = 200
OPT_XTOL = 1e-6
DEBUG_FOCK = True

def mpi_reduce(m, op=MPI.SUM, root=0):
    is_array = isinstance(m, np.ndarray)

    if size > 1:
        m_red = np.zeros_like(m)
        comm.Reduce(np.asarray(m), m_red, op=op, root=root)
        m = m_red
        comm.Bcast(m, root=root)

        if not is_array:
            m = m.ravel()[0]

    return m

def mpi_split(n):
    # i.e. n = 10, size = 4 -> (3, 3, 2, 2)
    lst = [n // size + int(n % size > x) for x in range(size)]
    assert sum(lst) == n
    return tuple(lst)

def _reorder_fortran(a, trans_a=False):
    if a.flags.c_contiguous:
        return np.array(a.T, copy=False, order='F'), not trans_a
    else:
        return np.array(a, copy=False, order='F'), trans_a

def _reorder_c(a, trans_a=False):
    if a.flags.f_contiguous:
        return np.array(a.T, copy=False, order='C'), not trans_a
    else:
        return np.array(a, copy=False, order='C'), trans_a

_is_contiguous = lambda a: a.flags.c_contiguous or a.flags.f_contiguous

def dgemm(a, b, c=None, alpha=1.0, beta=0.0):
    # Fortran memory layout so we reorder memory without copying, and then
    # form c^T, finally converting to C layout without copying.

    if not _is_contiguous(a) or not _is_contiguous(b):
        print('WARNING: DGEMM called on non-contiguous data')

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

    c = blas.dgemm(alpha=alpha, a=b, b=a, c=c, beta=beta, trans_a=not tb, trans_b=not ta)

    c, tc = _reorder_c(c)

    return c

dot = np.dot

def build_x(ixQ, Qja, nphys, nocc, nvir):
    # Builds the X array, entirely equivalent to the zeroth-order moment of the self-energy

    x = np.zeros((nphys, nphys))

    buf1 = np.zeros((nphys, nocc * nvir))
    buf2 = np.zeros((nocc * nphys, nvir))

    for i in range(rank, nocc, size):
        xja = dot(ixQ[i*nphys:(i+1)*nphys], Qja, out=buf1)
        xia = dot(ixQ, Qja[:,i*nvir:(i+1)*nvir], out=buf2)
        xia = reshape_internal(xia, (nocc, nphys, nvir), (0,1), (nphys, nocc*nvir))
        x = dgemm(xja, xja.T, alpha=2, beta=1, c=x)
        x = dgemm(xja, xia.T, alpha=-1, beta=1, c=x)

    # Reduce
    x = mpi_reduce(x)

    return x

def build_m(gf_occ, gf_vir, ixQ, Qja, binv):
    # Builds the M array, with contributions from blocks of auxiliaries

    nphys = gf_occ.nphys
    nocc = gf_occ.naux
    nvir = gf_vir.naux

    m = np.zeros((nphys, nphys))

    eo, ev = gf_occ.e, gf_vir.e
    indices = util.mpi.tril_indices_rows(nocc)
    pos_factor = np.sqrt(0.5)
    neg_factor = np.sqrt(1.5)

    for i in indices[rank]:
        xQ = ixQ[i*nphys:(i+1)*nphys]
        Qa = Qja[:,i*nvir:(i+1)*nvir]

        vija = dot(xQ, Qja[:,:i*nvir]).reshape((nphys, -1))
        vjia = dot(ixQ[:i*nphys], Qa)
        vjia = reshape_internal(vjia, (i, nphys, nvir), (0,1), (nphys, i*nvir))
        viia = dot(xQ, Qa)

        ea = eb = eo[i] + np.subtract.outer(eo[:i], ev).ravel()
        ec = 2 * eo[i] - ev

        va = neg_factor * (vija - vjia)
        vb = pos_factor * (vija + vjia)
        vc = viia

        qa = dot(binv.T, va)
        qb = dot(binv.T, vb)
        qc = dot(binv.T, vc)

        m = dgemm(qa * ea[None], qa.T, c=m, beta=1)
        m = dgemm(qb * eb[None], qb.T, c=m, beta=1)
        m = dgemm(qc * ec[None], qc.T, c=m, beta=1)

    # Reduce
    m = mpi_reduce(m)

    return m

def _get_df_blocks(eri, maxblk=DF_MAXBLK):
    # Get slices to iterate over blocks of the DF integrals in the current
    # process only.

    naux = eri.shape[0]
    blks = mpi_split(naux)

    start = sum(blks[:rank])
    stop = min(sum(blks[:rank+1]), naux)

    return [slice(start+i*maxblk, min(start+(i+1)*maxblk, stop)) 
            for i in range(blks[rank] // maxblk + 1)]


def df_ao2mo(eri, ci, cj, sym_in='s2', sym_out='s2', maxblk=DF_MAXBLK):
    # ao2mo for density fitted integrals with specific input and output symmetry
    
    naux = eri.shape[0]
    ijsym, nij, cij, sij = ao2mo.incore._conc_mos(ci, cj, compact=True)
    i, j = ci.shape[1], cj.shape[1]

    Qij = np.zeros((naux, i*(i+1)//2 if sym_out == 's2' else i*j))

    for s in _get_df_blocks(eri, maxblk):
        Qij[s] = ao2mo._ao2mo.nr_e2(eri[s], cij, sij, out=Qij[s],
                                    aosym=sym_in, mosym=sym_out)

    return Qij

def build_aux(gf_occ, gf_vir, eri):
    # Iterates the Green's function to directly build the compressed self-energy auxiliaries

    nphys = gf_occ.nphys
    nocc = gf_occ.naux
    nvir = gf_vir.naux

    ixQ = df_ao2mo(eri, gf_occ.v, np.eye(nphys), sym_in='s2', sym_out='s1').T
    Qja = df_ao2mo(eri, gf_occ.v, gf_vir.v, sym_in='s2', sym_out='s1')

    x = build_x(ixQ, Qja, nphys, nocc, nvir)
    b = np.linalg.cholesky(x).T
    binv = np.linalg.inv(b)
    m = build_m(gf_occ, gf_vir, ixQ, Qja, binv)

    e, c = util.eigh(m)
    c = dot(b.T, c[:nphys])
    se = gf_occ.new(e, c)

    return se

to_ptr = lambda m : m.ctypes.data_as(ctypes.c_void_p)
_fmmm = ao2mo._ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
_fdrv = ao2mo._ao2mo.libao2mo.AO2MOnr_e2_drv
_ftrans = ao2mo._ao2mo.libao2mo.AO2MOtranse2_nr_s2

def get_fock(rdm1, h1e, eri, maxblk=DF_MAXBLK):
    # Gets the Fock matrix (MO basis)
    # Transforms to AO basis and then back to appease pyscf

    nphys = rdm1.shape[-1]
    naux = eri.shape[0]
    
    j = np.zeros((nphys*(nphys+1)//2))
    k = np.zeros((nphys, nphys))

    rdm1tril = lib.pack_tril(rdm1 + np.tril(rdm1, k=-1))
    rargs = (ctypes.c_int(nphys), (ctypes.c_int*4)(0, nphys, 0, nphys), lib.c_null_ptr(), ctypes.c_int(0))
    buf = np.empty((2, maxblk, nphys, nphys))

    for s in _get_df_blocks(eri, maxblk):
        eri1 = eri[s]
        naux_block = eri1.shape[0]

        rho = dot(eri1, rdm1tril)
        j += dot(rho, eri1)

        buf1 = buf[0,:naux_block]
        _fdrv(_ftrans, _fmmm, to_ptr(buf1), to_ptr(eri1), to_ptr(rdm1), ctypes.c_int(naux_block), *rargs)

        buf2 = lib.unpack_tril(eri1, out=buf[1])
        k = dgemm(buf1.reshape(-1, nphys).T, buf2.reshape(-1, nphys), c=k, beta=1)

    j = mpi_reduce(j)
    k = mpi_reduce(k)

    j = lib.unpack_tril(j).reshape(rdm1.shape)
    k = k.reshape(rdm1.shape)

    f = h1e + j - 0.5 * k

    return f

def minimize(obj, bounds=(None, None), method='brent', maxiter=OPT_MAXITER, tol=OPT_XTOL, x0=None):
    # Runs the chemical potential minimization with a bunch of different
    # available methods, uses OpenMP on the root process only.
    # Best methods are golden and lstsq, golden uses more function calls with less
    # additional overhead, whereas lstsq uses fewer function calls and more overhead,
    # lstsq is best unless we efficiently parallelise the objective function.

    if method == 'brent':
        kwargs = dict(method='bounded', bounds=bounds, options=dict(maxiter=maxiter, xatol=tol))
        f = optimize.minimize_scalar
    elif method == 'golden':
        kwargs = dict(method='golden', bounds=bounds, options=dict(maxiter=maxiter, xtol=tol))
        f = optimize.minimize_scalar
    elif method == 'newton':
        kwargs = dict(method='TNC', bounds=[bounds], x0=x0, options=dict(maxiter=maxiter, xtol=tol))
        f = optimize.minimize
    elif method == 'bfgs':
        kwargs = dict(method='L-BFGS-B', bounds=[bounds], x0=x0, options=dict(maxiter=maxiter, ftol=tol))
        f = optimize.minimize
    elif method == 'lstsq':
        kwargs = dict(method='SLSQP', x0=x0, options=dict(maxiter=maxiter, ftol=tol))
        f = optimize.minimize
    elif method == 'stochastic':
        kwargs = dict(bounds=[(-5, 5)], maxiter=maxiter, tol=tol)
        f = optimize.differential_evolution
    else:
        raise ValueError

    opt = None

    if rank == 0:
        with lib.with_omp_threads(size):
            opt = f(obj, **kwargs)

    if size > 1:
        opt = comm.bcast(opt, root=0)

    return opt

def fock_loop_rhf(se, hf, rdm1, eri, debug=DEBUG_FOCK, opt_method='lstsq'):
    # Simple version of auxgf.agf2.fock.fock_loop_rhf

    def diag_fock_ext(cpt):
        se.as_hamiltonian(fock, chempot=cpt, out=buf)
        w, v = util.eigh(buf)
        cpt, err = util.chempot._find_chempot(hf.nao, hf.nelec, h=(w, v))
        return w, v, cpt, err

    diis = util.DIIS(8)
    h1e = hf.h1e_mo
    fock = get_fock(rdm1, h1e, eri)
    rdm1_prev = np.zeros_like(rdm1)

    obj = lambda x : abs((diag_fock_ext(x))[-1])
    buf = np.zeros((se.nphys + se.naux,)*2)
    w, v = se.eig(fock)

    if debug and rank == 0:
        print('%17s %17s %17s' % ('-'*17, '-'*17, '-'*17))
        print('%17s %17s %17s' % ('nelec'.center(17), 'chempot'.center(17), 'density'.center(17)))
        print('%4s %12s %4s %12s %4s %12s' % ('iter', 'error', 'iter', 'error', 'iter', 'error'))
        print('%17s %17s %17s' % ('-'*17, '-'*17, '-'*17))

    for niter1 in range(20):
        w, v, se.chempot, error = diag_fock_ext(0)

        hoqmo = np.max(w[w < se.chempot])
        luqmo = np.min(w[w >= se.chempot])

        opt = minimize(lambda x: abs((diag_fock_ext(x))[-1]), bounds=(hoqmo, luqmo), x0=se.chempot, method=opt_method)
        se._ener -= opt.x

        for niter2 in range(50):
            w, v, se.chempot, error = diag_fock_ext(0)

            v_phys_occ = v[:hf.nao, w < se.chempot]
            rdm1 = dot(v_phys_occ, v_phys_occ.T) * 2
            fock = get_fock(rdm1, h1e, eri)

            fock = diis.update(fock)
            derr = np.linalg.norm(rdm1 - rdm1_prev)
            if derr < 1e-6:
                break

            rdm1_prev = rdm1.copy()

        if debug and rank == 0:
            print('%4d %12.6g %4d %12.6g %4d %12.6g' % (niter1, error, opt.nfev, np.ravel(opt.fun)[0], niter2, derr))

        if derr < 1e-6 and abs(error) < 1e-6:
            break

    if debug and rank == 0:
        print('%17s %17s %17s' % ('-'*17, '-'*17, '-'*17))

    converged = derr < 1e-6 and abs(error) < 1e-6

    return se, rdm1, converged

def energy_2body_aux(gf, se):
    # MPI parallel version of aux.energy_2body_aux

    e2b = 0.0

    for l in range(rank, gf.nocc, size):
        vxl = gf.v[:,l]
        vxk = se.v[:,se.nocc:]

        dlk = 1.0 / (gf.e[l] - se.e[se.nocc:])

        e2b += util.einsum('xk,yk,x,y,k->', vxk, vxk, vxl, vxl, dlk)

    e2b = 2.0 * np.ravel(e2b)[0]
    e2b = mpi_reduce(e2b)

    return e2b

def run(rhf, maxiter=20, etol=1e-6):
    eri = rhf._pyscf.with_df._cderi
    eri = df_ao2mo(eri, rhf.c, rhf.c, sym_in='s2', sym_out='s2')

    rdm1 = rhf.rdm1_mo
    gf = aux.Aux(rhf.e, np.eye(rhf.nao), chempot=rhf.chempot)
    se = aux.Aux([], [[],]*rhf.nao, chempot=rhf.chempot)
    e_tot = rhf.e_tot

    for i in range(maxiter):
        se, rdm1, conv = fock_loop_rhf(se, rhf, rdm1, eri)
        
        e, c = se.eig(rhf.get_fock(rdm1, basis='mo'))
        gf = se.new(e, c[:rhf.nao])

        se  = build_aux(gf.as_occupied(), gf.as_virtual(), eri)
        se += build_aux(gf.as_virtual(),  gf.as_occupied(), eri)

        e_tot_prev = e_tot
        e_1body = rhf.energy_1body(rhf.h1e_mo, rdm1, rhf.get_fock(rdm1, basis='mo'))
        e_1body += rhf.mol.e_nuc
        e_2body = energy_2body_aux(gf, se)
        e_tot = e_1body + e_2body

        if rank == 0:
            print('E(tot) = %14.8f' % e_tot)

        if abs(e_tot - e_tot_prev) < etol:
            break


if __name__ == '__main__':
    m = mol.Molecule(atoms='O 0 0 0; O 0 0 1', basis='aug-cc-pvdz')
    rhf = hf.RHF(m, with_df=True).run()

    if 0:
        run(rhf)

    if 0:
        rhf = hf.RHF(m).run()
        gf2 = agf2.RAGF2(rhf, nmom=(1,0), verbose=False)
        gf2.run()
        print(gf2.e_tot)

    if 1:
        import IPython
        ipython = IPython.get_ipython()
        ipython.magic('load_ext line_profiler')
        ipython.magic('lprun -f build_m run(rhf, maxiter=5)')






























