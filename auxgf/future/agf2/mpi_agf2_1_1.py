from auxgf import *
import numpy as np
from mpi4py import MPI
from pyscf import lib, ao2mo, df
from scipy.optimize import minimize_scalar
from scipy.linalg import blas

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# The density fitting procedure here can be made to be more efficient,
# i.e. iterating over blocks of the DF auxiliary basis and also doing
# basis-transformations in situ, but this is still be O(n^3) in memory

# Reshape pattern for shitty exchange interactions
ab_c__to__b_ac = lambda x,a,b,c : x.reshape((a,b,c)).swapaxes(0,1).reshape((b, a*c))

def build_x(ixQ, Qja, nphys, nocc, nvir):
    # Builds the X array, entirely equivalent to the zeroth-order moment of the self-energy

    x = np.zeros((nphys, nphys))

    for i in range(rank, nocc, size):
        xja = np.dot(ixQ[i*nphys:(i+1)*nphys], Qja)
        xia = np.dot(ixQ, Qja[:,i*nvir:(i+1)*nvir]).reshape((nocc, nphys, nvir)).swapaxes(0,1).reshape((nphys, -1))
        x += np.dot(xja, xja.T) * 2
        x -= np.dot(xja, xia.T)

    # Reduce
    if size > 1:
        x_red = np.zeros_like(x)
        comm.Reduce(x, x_red, op=MPI.SUM, root=0)
        x = x_red
        comm.Bcast(x, root=0)

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
    vjia = np.dot(ixQ[:i*nphys], Qa).reshape((i, nphys, nvir)).swapaxes(0,1).reshape((nphys, -1))
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
    if size > 1:
        m_red = np.zeros_like(m)
        comm.Reduce(m, m_red, op=MPI.SUM, root=0)
        m = m_red
        comm.Bcast(m, root=0)

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

def fock_loop_rhf(se, hf, rdm1):
    # Simple version of auxgf.agf2.fock.fock_loop_rhf

    diis = util.DIIS(8)
    fock = hf.get_fock(rdm1, basis='mo')
    rdm1_prev = np.zeros_like(rdm1)

    def diag_fock_ext(cpt):
        w, v = se.eig(fock, chempot=cpt)
        cpt, err = util.find_chempot(hf.nao, hf.nelec, h=(w, v)) # simple aufbau ting
        return w, v, cpt, err

    obj = lambda x : abs((diag_fock_ext(x))[-1])
    minimize = lambda bounds : minimize_scalar(obj, bounds=bounds, method='bounded', options={'maxiter': 1000, 'xatol': 1e-6})

    for niter1 in range(20):
        w, v, se.chempot, error = diag_fock_ext(0)

        hoqmo = np.max(w[w < se.chempot])
        luqmo = np.min(w[w >= se.chempot])

        se._ener -= minimize((hoqmo, luqmo)).x

        for niter2 in range(50):
            w, v, se.chempot, error = diag_fock_ext(0)

            v_phys_occ = v[:hf.nao, w < se.chempot]
            rdm1 = np.dot(v_phys_occ, v_phys_occ.T) * 2
            fock = hf.get_fock(rdm1, basis='mo')

            fock = diis.update(fock)
            derr = np.linalg.norm(rdm1 - rdm1_prev)
            if derr < 1e-6:
                break

            rdm1_prev = rdm1.copy()

        if derr < 1e-6 and abs(error) < 1e-6:
            break

    converged = derr < 1e-6 and abs(error) < 1e-6

    return se, rdm1, converged

def run(rhf):
    eri = rhf._pyscf.with_df._cderi
    eri = df_ao2mo(eri, rhf.c, rhf.c, sym_in='s2', sym_out='s2')

    rdm1 = rhf.rdm1_mo
    gf = aux.Aux(rhf.e, np.eye(rhf.nao), chempot=rhf.chempot)
    se = aux.Aux([], [[],]*rhf.nao, chempot=rhf.chempot)
    e_tot = rhf.e_tot

    for i in range(2):
        se, rdm1, conv = fock_loop_rhf(se, rhf, rdm1)
        
        e, c = se.eig(rhf.get_fock(rdm1, basis='mo'))
        gf = se.new(e, c[:rhf.nao])

        se  = build_aux(gf.as_occupied(), gf.as_virtual(), eri)
        se += build_aux(gf.as_virtual(),  gf.as_occupied(), eri)

        e_tot_prev = e_tot
        e_1body = rhf.energy_1body(rhf.h1e_mo, rdm1, rhf.get_fock(rdm1, basis='mo'))
        e_1body += rhf.mol.e_nuc
        e_2body = aux.energy_2body_aux(gf, se)
        e_tot = e_1body + e_2body

        if rank == 0:
            print(e_1body, e_2body, e_tot)

        if abs(e_tot - e_tot_prev) < 1e-6:
            break


if __name__ == '__main__':
    m = mol.Molecule(atoms='O 0 0 0; O 0 0 1', basis='aug-cc-pvdz')
    rhf = hf.RHF(m, with_df=True).run()
    run(rhf)

    #import IPython
    #ipython = IPython.get_ipython()
    #ipython.magic('load_ext line_profiler')
    #ipython.magic('lprun -f run run(rhf)')





























