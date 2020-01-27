import numpy as np

from auxgf import mol, hf, aux, util, grids


rhf = hf.RHF(mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='sto3g'))

eri = rhf.eri_mo
e = rhf.e
cpt = rhf.chempot
o = slice(None, np.sum(e < cpt))
v = slice(np.sum(e < cpt), None)
imfq = grids.ImFqGrid(2**5, beta=2**3)

xija = eri[:,o,o,v]
xabi = eri[:,v,v,o]
eija = util.outer_sum([e[o], e[o], -e[v]])
eabi = util.outer_sum([e[v], e[v], -e[o]])

se_mp2_ref  = np.einsum('xija,yija,wija->wxy', xija, 2*xija-xija.swapaxes(1,2), 1.0/(1.0j*imfq[:,None,None,None]-(eija-cpt)[None]))
se_mp2_ref += np.einsum('xija,yija,wija->wxy', xabi, 2*xabi-xabi.swapaxes(1,2), 1.0/(1.0j*imfq[:,None,None,None]-(eabi-cpt)[None]))


se_mp2_aux = aux.build_rmp2(e, eri, chempot=cpt)
se_mp2 = se_mp2_aux.as_spectrum(imfq)
assert np.allclose(se_mp2_ref, se_mp2)


se_mp2 = aux.build_rmp2_se_direct(e, eri, imfq, chempot=cpt)
assert np.allclose(se_mp2_ref, se_mp2)


aux_empty = aux.Aux([], np.zeros((rhf.nao, 0)), chempot=cpt)
se_mp2_aux = aux.build_rmp2_iter(aux_empty, rhf.fock_mo, eri)
se_mp2 = se_mp2_aux.as_spectrum(imfq)
assert np.allclose(se_mp2_ref, se_mp2)
