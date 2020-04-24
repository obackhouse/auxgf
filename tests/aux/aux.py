import numpy as np
import os

from auxgf import aux, grids, util


nphys = 4
naux = 20

h = np.dot(*([2 * np.random.random((nphys, nphys)) - 1,]*2))
e = 2 * np.random.random((naux)) - 1
v = 2 * np.random.random((nphys, naux)) - 1
vec = 2 * np.random.random((nphys + naux, 2)) - 1
cpt = 0.1

se = aux.Aux(e, v, chempot=cpt)

assert se.chempot == cpt
assert np.allclose(se.e, e)
assert np.allclose(se.v, v)
assert np.allclose(se.e_occ, e[e < cpt])
assert np.allclose(se.v_occ, v[:,e < cpt])

imfq = grids.ImFqGrid(2**8, beta=2**5)
refq = grids.ReFqGrid(2**8, eta=0.1, minpt=-5, maxpt=5)
se_imfq_ref = np.einsum('xk,yk,wk->wxy', v, v, 1.0 / util.outer_sum([1.0j * imfq, -(e-cpt)]))
se_refq_ref = np.einsum('xk,yk,wk->wxy', v, v, 1.0 / util.outer_sum([refq, -(e-cpt) + np.sign(e-cpt) * 1j * refq.eta]))
se_imfq = se.as_spectrum(imfq)
se_refq = se.as_spectrum(refq)
assert np.allclose(se_imfq, se_imfq_ref)
assert np.allclose(se_refq, se_refq_ref)
del se_refq, se_imfq_ref, se_refq_ref

imfq = grids.ImFqGrid(2**8, beta=2**5)
refq = grids.ReFqGrid(2**8, eta=0.1, minpt=-5, maxpt=5)
dse_imfq_ref = -np.einsum('xk,yk,wk->wxy', v, v, 1.0 / util.outer_sum([1.0j * imfq, -(e-cpt)])**2)
dse_refq_ref = -np.einsum('xk,yk,wk->wxy', v, v, 1.0 / util.outer_sum([refq, -(e-cpt) + np.sign(e-cpt) * 1j * refq.eta])**2)
dse_imfq = se.as_derivative(imfq)
dse_refq = se.as_derivative(refq)
assert np.allclose(dse_imfq, dse_imfq_ref)
assert np.allclose(dse_refq, dse_refq_ref)
del dse_refq, dse_imfq, dse_imfq_ref, dse_refq_ref, refq

ham_ref = np.block([[h, v], [v.T, np.diag(e)]])
ham = se.as_hamiltonian(h)
assert np.allclose(ham, ham_ref)

rot_ref = np.dot(ham, vec)
rot = se.dot(h, vec)
assert np.allclose(rot, rot_ref)
del rot, rot_ref

w_ref, l_ref = se.eig(h)
w, l = np.linalg.eigh(ham_ref)
assert np.allclose(w, w_ref)
assert np.allclose(l, l_ref)
del ham, ham_ref, w, l, w_ref, l_ref

e_window_ref = e[np.logical_and(e >= 0.25, e < 0.4)]
e_window = se.as_window(0.25, 0.4).e
assert np.allclose(e_window, e_window_ref)
del e_window, e_window_ref

moms_ref = np.einsum('xk,yk,nk->nxy', v, v, e[None] ** np.arange(5)[:,None])
moms = se.moment(np.arange(5))
assert np.allclose(moms_ref, moms)
del moms

#red = se.merge()
#assert np.allclose(np.sum(se.v**2), np.sum(red.v**2))
#del red
#
#red = se.se_compress(h, 4)
#se_imfq = red.as_spectrum(imfq)
#assert np.allclose(se_imfq, se_imfq_ref)
#del se_imfq
#
#red = se.gf_compress(h, 4)
#se_imfq = red.as_spectrum(imfq)
#assert np.allclose(se_imfq, se_imfq_ref)
#del se_imfq
#
#red = se.compress(h, (3,4))
#se_imfq = red.as_spectrum(imfq)
#assert np.allclose(se_imfq, se_imfq_ref)
#del se_imfq, se_imfq_ref

if util.mpi.size == 1:  # FIXME
    se.save('se.dat')
    se_load = se.load('se.dat')
    se_copy = se.copy()
    assert se == se_load and se == se_copy
    os.remove('se.dat')

