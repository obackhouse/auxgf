import numpy as np

from auxgf import mol, hf, aux, util, grids


# if copying from this file, note that this only builds alpha self-energy


uhf = hf.UHF(mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='sto3g')).run()

eri = uhf.eri_mo
e = uhf.e
cpt = uhf.chempot
o = (slice(None, np.sum(e[0] < cpt[0])), slice(None, np.sum(e[1] < cpt[1])))
v = (slice(np.sum(e[0] < cpt[0]), None), slice(np.sum(e[1] < cpt[1]), None))
imfq = grids.ImFqGrid(2**5, beta=2**3)

xija = (eri[0,0][:,o[0],o[0],v[0]], eri[0,1][:,o[0],o[1],v[1]])
xabi = (eri[0,0][:,v[0],v[0],o[0]], eri[0,1][:,v[0],v[1],o[1]])
eija = (util.outer_sum([e[0][o[0]], e[0][o[0]], -e[0][v[0]]]), util.outer_sum([e[0][o[0]], e[1][o[1]], -e[1][v[1]]]))
eabi = (util.outer_sum([e[0][v[0]], e[0][v[0]], -e[0][o[0]]]), util.outer_sum([e[0][v[0]], e[1][v[1]], -e[1][o[1]]]))

se_mp2_ref  = np.einsum('xija,yija,wija->wxy', xija[0], xija[0]-xija[0].swapaxes(1,2), 1.0/(1.0j*imfq[:,None,None,None]-(eija[0]-cpt[0])[None]))
se_mp2_ref += np.einsum('xija,yija,wija->wxy', xija[1], xija[1],                       1.0/(1.0j*imfq[:,None,None,None]-(eija[1]-cpt[1])[None]))
se_mp2_ref += np.einsum('xabi,yabi,wabi->wxy', xabi[0], xabi[0]-xabi[0].swapaxes(1,2), 1.0/(1.0j*imfq[:,None,None,None]-(eabi[0]-cpt[0])[None]))
se_mp2_ref += np.einsum('xabi,yabi,wabi->wxy', xabi[1], xabi[1],                       1.0/(1.0j*imfq[:,None,None,None]-(eabi[1]-cpt[1])[None]))


se_mp2_aux_a = aux.build_ump2((e[0], e[1]), (eri[0,0], eri[0,1]), chempot=cpt)
se_mp2 = se_mp2_aux_a.as_spectrum(imfq)
assert np.allclose(se_mp2_ref, se_mp2)


se_mp2 = np.sum([x.as_spectrum(imfq) for x in aux.build_ump2_direct((e[0], e[1]), (eri[0,0], eri[0,1]), chempot=cpt)], axis=0)
assert np.allclose(se_mp2_ref, se_mp2)


se_mp2 = aux.build_ump2_se_direct((e[0], e[1]), (eri[0,0], eri[0,1]), imfq, chempot=cpt)
assert np.allclose(se_mp2_ref, se_mp2)


aux_empty_a = aux.Aux([], np.zeros((uhf.nao, 0)), chempot=cpt[0])
aux_empty_b = aux.Aux([], np.zeros((uhf.nao, 0)), chempot=cpt[1])
se_mp2_aux = aux.build_ump2_iter((aux_empty_a, aux_empty_b), uhf.fock_mo, eri)[0]
se_mp2 = se_mp2_aux.as_spectrum(imfq)
assert np.allclose(se_mp2_ref, se_mp2, atol=1e-07)



uhf = hf.UHF(mol.Molecule(atoms='H 0 0 0; Be 0 0 1.00', basis='sto3g', spin=1)).run()

eri = uhf.eri_mo
e = uhf.e
cpt = uhf.chempot
o = (slice(None, np.sum(e[0] < cpt[0])), slice(None, np.sum(e[1] < cpt[1])))
v = (slice(np.sum(e[0] < cpt[0]), None), slice(np.sum(e[1] < cpt[1]), None))
imfq = grids.ImFqGrid(2**5, beta=2**3)

xija = (eri[0,0][:,o[0],o[0],v[0]], eri[0,1][:,o[0],o[1],v[1]])
xabi = (eri[0,0][:,v[0],v[0],o[0]], eri[0,1][:,v[0],v[1],o[1]])
eija = (util.outer_sum([e[0][o[0]], e[0][o[0]], -e[0][v[0]]]), util.outer_sum([e[0][o[0]], e[1][o[1]], -e[1][v[1]]]))
eabi = (util.outer_sum([e[0][v[0]], e[0][v[0]], -e[0][o[0]]]), util.outer_sum([e[0][v[0]], e[1][v[1]], -e[1][o[1]]]))

se_mp2_ref  = np.einsum('xija,yija,wija->wxy', xija[0], xija[0]-xija[0].swapaxes(1,2), 1.0/(1.0j*imfq[:,None,None,None]-(eija[0]-cpt[0])[None]))
se_mp2_ref += np.einsum('xija,yija,wija->wxy', xija[1], xija[1],                       1.0/(1.0j*imfq[:,None,None,None]-(eija[1]-cpt[0])[None]))
se_mp2_ref += np.einsum('xabi,yabi,wabi->wxy', xabi[0], xabi[0]-xabi[0].swapaxes(1,2), 1.0/(1.0j*imfq[:,None,None,None]-(eabi[0]-cpt[0])[None]))
se_mp2_ref += np.einsum('xabi,yabi,wabi->wxy', xabi[1], xabi[1],                       1.0/(1.0j*imfq[:,None,None,None]-(eabi[1]-cpt[0])[None]))


se_mp2_aux_a = aux.build_ump2((e[0], e[1]), (eri[0,0], eri[0,1]), chempot=cpt)
se_mp2 = se_mp2_aux_a.as_spectrum(imfq)
assert np.allclose(se_mp2_ref, se_mp2)


se_mp2 = aux.build_ump2_se_direct((e[0], e[1]), (eri[0,0], eri[0,1]), imfq, chempot=cpt)
assert np.allclose(se_mp2_ref, se_mp2)


aux_empty_a = aux.Aux([], np.zeros((uhf.nao, 0)), chempot=cpt[0])
aux_empty_b = aux.Aux([], np.zeros((uhf.nao, 0)), chempot=cpt[1])
se_mp2_aux = aux.build_ump2_iter((aux_empty_a, aux_empty_b), uhf.fock_mo, eri)[0]
se_mp2 = se_mp2_aux.as_spectrum(imfq)
assert np.allclose(se_mp2_ref, se_mp2)
