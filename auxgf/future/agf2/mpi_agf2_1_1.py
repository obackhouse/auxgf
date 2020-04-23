from auxgf import *
import numpy as np
from mpi4py import MPI

# The density fitting procedure here can be made to be more efficient,
# i.e. iterating over blocks of the DF auxiliary basis and also doing
# basis-transformations in situ, but this is still be O(n^3) in memory

def build_x(ixQ, Qja):
    # Builds the X array, entirely equivalent to the zeroth-order moment of the self-energy

    x = np.zeros((rhf.nao, rhf.nao))

    # This won't be split very equally (indices are tril)
    for i in range(rank, ixQ.shape[0], size):
        xja = util.einsum('xQ,Qja->xja', ixQ[i], Qja)
        xia = util.einsum('ixQ,Qa->xia', ixQ, Qja[:,i])
        x += util.einsum('xja,yja->xy', xja, xja) * 2
        x -= util.einsum('xja,yja->xy', xja, xia)

    # Reduce
    if size > 1:
        x_red = np.zeros_like(x)
        comm.Reduce(x, x_red, op=MPI.SUM, root=0)
        x = x_red
        comm.Bcast(x, root=0)

    return x

def build_rmp2_part_direct(eo, ev, ixQ, Qja, i=0):
    # Builds all (i,j,a) auxiliaries for a single index i using density fitted integrals

    _, nphys, nocc = ixQ.shape
    nvir = ev.size

    pos_factor = np.sqrt(0.5)
    neg_factor = np.sqrt(1.5)

    nja = i * nvir

    e = np.zeros((nja*2+nvir))
    v = np.zeros((nphys, nja*2+nvir))

    m1 = slice(None, nja)
    m2 = slice(nja, 2*nja)
    m3 = slice(2*nja, 2*nja+nvir)

    vija = util.einsum('xQ,Qja->xja', ixQ[i], Qja[:,:i]).reshape((nphys, nja)) 
    vjia = util.einsum('ixQ,Qa->xia', ixQ[:i], Qja[:,i]).reshape((nphys, nja))
    viia = util.einsum('xQ,Qa->xa', ixQ[i], Qja[:,i])

    e[m1] = e[m2] = eo[i] + np.subtract.outer(eo[:i], ev).flatten()
    e[m3] = 2 * eo[i] - ev

    v[:,m1] = neg_factor * (vija - vjia)
    v[:,m2] = pos_factor * (vija + vjia)
    v[:,m3] = viia

    return e, v

def build_m(gf_occ, gf_vir, ixQ, Qja, binv):
    # Builds the M array

    m = np.zeros((rhf.nao, rhf.nao))

    for i in range(rank, ixQ.shape[0], size):
        e, v = build_rmp2_part_direct(gf_occ.e, gf_vir.e, ixQ, Qja, i=i)
        q = np.dot(binv.T, v)
        m += util.einsum('xk,k,yk->xy', q, e, q)

    # Reduce
    if size > 1:
        m_red = np.zeros_like(m)
        comm.Reduce(m, m_red, op=MPI.SUM, root=0)
        m = m_red
        comm.Bcast(m, root=0)

    return m

def build_aux(gf_occ, gf_vir):
    # Iterates the Green's function to directly build the compressed self-energy auxiliaries

    ixQ = util.einsum('Qxy,yi->ixQ', eri, gf_occ.v)  # should have better memory efficiency
    Qja = util.einsum('Qxy,xj,ya->Qja', eri, gf_occ.v, gf_vir.v)

    x = build_x(ixQ, Qja)
    b = np.linalg.cholesky(x).T
    binv = np.linalg.inv(b)
    m = build_m(gf_occ, gf_vir, ixQ, Qja, binv)

    e, c = util.eigh(m)
    c = np.dot(b.T, c[:rhf.nao])
    se = gf_occ.new(e, c)

    return se

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    m = mol.Molecule(atoms='O 0 0 0; O 0 0 1', basis='aug-cc-pvdz')
    rhf = hf.RHF(m, with_df=True).run()

    eri = rhf.eri_mo  # Cholesky decomposed ERI tensor
    gf = aux.Aux(rhf.e, np.eye(rhf.nao), chempot=rhf.chempot)
    se = aux.Aux([], [[],]*rhf.nao, chempot=rhf.chempot)
    e_tot = rhf.e_tot

    while True:
        se, rdm1, conv = agf2.fock.fock_loop_rhf(se, rhf, rhf.rdm1_mo, verbose=False)
        
        e, c = se.eig(rhf.get_fock(rdm1, basis='mo'))
        gf = se.new(e, c[:rhf.nao])

        se  = build_aux(gf.as_occupied(), gf.as_virtual())
        se += build_aux(gf.as_virtual(),  gf.as_occupied())

        e_tot_prev = e_tot
        e_1body = rhf.energy_1body(rhf.h1e_mo, rdm1, rhf.get_fock(rdm1, basis='mo'))
        e_1body += rhf.mol.e_nuc
        e_2body = aux.energy_2body_aux(gf, se)
        e_tot = e_1body + e_2body

        if rank == 0:
            print(e_1body, e_2body, e_tot)

        if abs(e_tot - e_tot_prev) < 1e-6:
            break





























