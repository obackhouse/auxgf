from auxgf import *
import numpy as np
from mpi4py import MPI
from pyscf import lib, ao2mo, df
from pyscf.scf import jk
from pyscf.df import df_jk
from scipy import optimize
from scipy.linalg import blas
from collections import namedtuple
import ctypes

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# The density fitting procedure here can be made to be more efficient,
# i.e. iterating over blocks of the DF auxiliary basis and also doing
# basis-transformations in situ, but this is still be O(n^3) in memory

reshape_internal = lambda x, s1, swap, s2 : x.reshape(s1).swapaxes(*swap).reshape(s2)

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

to_ptr = lambda m : m.ctypes.data_as(ctypes.c_void_p)

_fmmm = ao2mo._ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
_fdrv = ao2mo._ao2mo.libao2mo.AO2MOnr_e2_drv
_ftrans = ao2mo._ao2mo.libao2mo.AO2MOtranse2_nr_s2

def fdrv(buf, eri, rdm1):
    # Some pyscf function for the Fock exchange interaction

    nphys = rdm1.shape[-1]
    naux = eri.shape[0]

    buf1 = buf[0,:naux]

    rargs = (ctypes.c_int(nphys), (ctypes.c_int*4)(0, nphys, 0, nphys), lib.c_null_ptr(), ctypes.c_int(0))
    _fdrv(_ftrans, _fmmm, to_ptr(buf1), to_ptr(eri), to_ptr(rdm1), ctypes.c_int(naux), *rargs)

    return buf1


def build_x(ixQ, Qja, nphys, nocc, nvir):
    # Builds the X array, entirely equivalent to the zeroth-order moment of the self-energy

    x = np.zeros((nphys, nphys))

    for i in range(rank, nocc, size):
        xja = np.dot(ixQ[i*nphys:(i+1)*nphys], Qja)
        xia = np.dot(ixQ, Qja[:,i*nvir:(i+1)*nvir])
        xia = reshape_internal(xia, (nocc, nphys, nvir), (0,1), (nphys, nocc*nvir))
        x += np.dot(xja, xja.T) * 2
        x -= np.dot(xja, xia.T)

    # Reduce
    x = mpi_reduce(x)

    return x

def build_rmp2_part_direct(eo, ev, ixQ, Qja, nphys, nocc, nvir, i=0):
    # Builds all (i,j,a) auxiliaries for a single index i using density fitted integrals

    pos_factor = np.sqrt(0.5)
    neg_factor = np.sqrt(1.5)

    nja = i * nvir

    e = np.zeros((nja*2+nvir))
    v = np.zeros((nphys, nja*2+nvir))

    m1 = slice(None, nja)
    m2 = slice(nja, 2*nja)
    m3 = slice(2*nja, 2*nja+nvir)

    xQ = ixQ[i*nphys:(i+1)*nphys]
    Qa = Qja[:,i*nvir:(i+1)*nvir]

    vija = np.dot(xQ, Qja[:,:i*nvir]).reshape((nphys, -1))
    vjia = np.dot(ixQ[:i*nphys], Qa)
    vjia = reshape_internal(vjia, (i, nphys, nvir), (0,1), (nphys, nja))
    viia = np.dot(xQ, Qa)

    e[m1] = e[m2] = eo[i] + np.subtract.outer(eo[:i], ev).ravel()
    e[m3] = 2 * eo[i] - ev

    v[:,m1] = neg_factor * (vija - vjia)
    v[:,m2] = pos_factor * (vija + vjia)
    v[:,m3] = viia

    return e, v

def build_m(gf_occ, gf_vir, ixQ, Qja, binv):
    # Builds the M array

    nphys = gf_occ.nphys
    nocc = gf_occ.naux
    nvir = gf_vir.naux

    m = np.zeros((nphys, nphys))

    indices = util.mpi.tril_indices_rows(nocc)
    for i in indices[rank]:
        e, v = build_rmp2_part_direct(gf_occ.e, gf_vir.e, ixQ, Qja, nphys, nocc, nvir, i=i)
        q = np.dot(binv.T, v)
        m += np.dot(q * e[None], q.T)

    # Reduce
    m = mpi_reduce(m)

    return m

def df_ao2mo(eri, ci, cj, sym_in='s2', sym_out='s2'):
    # ao2mo for density fitted integrals with specific input and output symmetry
    
    naux = eri.shape[0]
    ijsym, nij, cij, sij = ao2mo.incore._conc_mos(ci, cj, compact=True)
    i, j = ci.shape[1], cj.shape[1]

    Qij = np.zeros((naux, i*(i+1)//2 if sym_out == 's2' else i*j))
    # should we do this in blocks?
    Qij = ao2mo._ao2mo.nr_e2(eri, cij, sij, aosym=sym_in, mosym=sym_out, out=Qij)

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
    c = np.dot(b.T, c[:nphys])
    se = gf_occ.new(e, c)

    return se

def get_fock(rdm1, h1e, eri, max_memory=4000, blockdim=240):
    # Gets the Fock matrix (MO basis)
    # Transforms to AO basis and then back to appease pyscf

    nphys = rdm1.shape[-1]
    naux = eri.shape[0]
    
    j = np.zeros((nphys*(nphys+1)//2))
    k = np.zeros((nphys, nphys))

    rdm1tril = lib.pack_tril(rdm1 + np.tril(rdm1, k=-1))

    max_memory = max_memory - lib.current_memory()[0]
    blksize = max(4, int(min(blockdim, max_memory*.22e6/8/nphys**2)))
    mpi_blks = mpi_split(blksize)
    buf = np.empty((2, mpi_blks[rank], nphys, nphys))

    for i in range(0, naux, blksize):
        s = slice(i+sum(mpi_blks[:rank]), min(i+sum(mpi_blks[:rank+1]), naux))
        eri1 = eri[s]

        rho = np.dot(eri1, rdm1tril)
        j += np.dot(rho, eri1)

        buf1 = fdrv(buf, eri1, rdm1)
        buf2 = lib.unpack_tril(eri1, out=buf[1])
        k += np.dot(buf1.reshape(-1, nphys).T, buf2.reshape(-1, nphys))

    j = mpi_reduce(j)
    k = mpi_reduce(k)

    j = lib.unpack_tril(j).reshape(rdm1.shape)
    k = k.reshape(rdm1.shape)

    f = h1e + j - 0.5 * k

    return f

def minimize(obj, bounds, method='brent', maxiter=200, tol=1e-6):
    # Runs the chemical potential minimization with a bunch of different
    # available methods, uses OpenMP on the root process only.

    if method == 'brent':
        kwargs = dict(method='bounded', bounds=bounds, options=dict(maxiter=maxiter, xatol=tol))
        f = optimize.minimize_scalar
    elif method == 'golden':
        kwargs = dict(method='golden', bounds=bounds, options=dict(maxiter=maxiter, xtol=tol))
        f = optimize.minimize_scalar
    elif method == 'newton':
        kwargs = dict(method='TNC', bounds=[bounds], x0=[sum(bounds)/2], options=dict(maxiter=maxiter, xtol=tol))
        f = optimize.minimize
    elif method == 'bfgs':
        kwargs = dict(method='L-BFGS-B', bounds=[bounds], x0=[sum(bounds)/2], options=dict(maxiter=maxiter, ftol=tol))
        f = optimize.minimize
    elif method == 'lstsq':
        kwargs = dict(method='SLSQP', bounds=[bounds], x0=[sum(bounds)/2], options=dict(maxiter=maxiter, ftol=tol))
        f = optimize.minimize
    else:
        raise ValueError

    opt = None

    if rank == 0:
        with lib.with_omp_threads(size):
            opt = f(obj, **kwargs)

    if size > 1:
        opt = comm.bcast(opt, root=0)

    return opt

def fock_loop_rhf(se, hf, rdm1, eri, debug=True):
    # Simple version of auxgf.agf2.fock.fock_loop_rhf

    def diag_fock_ext(cpt):
        w, v = se.eig(fock, chempot=cpt)
        #cpt, err = util.find_chempot(hf.nao, hf.nelec, h=(w, v))
        cpt, err = util.chempot._find_chempot(hf.nao, hf.nelec, h=(w, v))
        return w, v, cpt, err

    diis = util.DIIS(8)
    h1e = hf.h1e_mo
    fock = get_fock(rdm1, h1e, eri)
    #fock = hf.get_fock(rdm1, basis='mo')
    rdm1_prev = np.zeros_like(rdm1)

    obj = lambda x : abs((diag_fock_ext(x))[-1])

    if debug and rank == 0:
        print('%17s %17s %17s' % ('-'*17, '-'*17, '-'*17))
        print('%17s %17s %17s' % ('nelec'.center(17), 'chempot'.center(17), 'density'.center(17)))
        print('%4s %12s %4s %12s %4s %12s' % ('iter', 'error', 'iter', 'error', 'iter', 'error'))
        print('%17s %17s %17s' % ('-'*17, '-'*17, '-'*17))

    for niter1 in range(20):
        w, v, se.chempot, error = diag_fock_ext(0)

        hoqmo = np.max(w[w < se.chempot])
        luqmo = np.min(w[w >= se.chempot])

        opt = minimize(lambda x: abs((diag_fock_ext(x))[-1]), (hoqmo, luqmo), method='golden')
        se._ener -= opt.x

        for niter2 in range(50):
            w, v, se.chempot, error = diag_fock_ext(0)

            v_phys_occ = v[:hf.nao, w < se.chempot]
            rdm1 = np.dot(v_phys_occ, v_phys_occ.T) * 2
            fock = get_fock(rdm1, h1e, eri)
            #fock = hf.get_fock(rdm1, basis='mo')

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
    run(rhf)

    if 0:
        rhf = hf.RHF(m).run()
        gf2 = agf2.RAGF2(rhf, nmom=(1,0), verbose=False)
        gf2.run()
        print(gf2.e_tot)

    if 0:
        import IPython
        ipython = IPython.get_ipython()
        ipython.magic('load_ext line_profiler')
        ipython.magic('lprun -f run run(rhf)')






























