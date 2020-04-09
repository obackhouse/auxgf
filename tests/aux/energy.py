import numpy as np

from auxgf import mol, hf, mp, aux, grids, util


rhf = hf.RHF(mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='sto3g')).run()
mp2 = mp.MP2(rhf).run()

gf_mo = aux.Aux(rhf.e, np.eye(rhf.nao), chempot=rhf.chempot)

se_mp2 = aux.build_mp2(rhf.e, rhf.eri_mo, chempot=rhf.chempot)
e_mp2_a = aux.energy_mp2_aux(rhf.e, se_mp2)
e_mp2_b = aux.energy_mp2_aux(rhf.e, se_mp2, both_sides=True)
e_mp2_c = aux.energy_2body_aux(gf_mo, se_mp2) / 2
e_mp2_d = aux.energy_2body_aux(gf_mo, se_mp2, both_sides=True) / 2
assert np.allclose(mp2.e_corr, [e_mp2_a, e_mp2_b, e_mp2_c, e_mp2_d])


uhf = hf.UHF(mol.Molecule(atoms='H 0 0 0; Be 0 0 1.00', basis='sto3g', spin=1)).run()
mp2 = mp.MP2(uhf).run()

gf_mo = (aux.Aux(uhf.e[0], np.eye(uhf.nao), chempot=uhf.chempot[0]), aux.Aux(uhf.e[1], np.eye(uhf.nao), chempot=uhf.chempot[1]))

se_mp2 = aux.build_mp2(uhf.e, uhf.eri_mo, chempot=uhf.chempot)
e_mp2_a = aux.energy_mp2_aux(uhf.e, se_mp2)
e_mp2_b = aux.energy_mp2_aux(uhf.e, se_mp2, both_sides=True)
e_mp2_c = aux.energy_2body_aux(gf_mo, se_mp2) / 2
e_mp2_d = aux.energy_2body_aux(gf_mo, se_mp2, both_sides=True) / 2
assert np.allclose(mp2.e_corr, [e_mp2_a, e_mp2_b, e_mp2_c, e_mp2_d])
