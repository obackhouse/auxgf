import numpy as np

from auxgf import mol, hf, mp


rhf = hf.RHF(mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='sto3g')).run()
mp2 = mp.MP2(rhf).run()
oomp2 = mp.OOMP2(rhf).run()
assert abs(mp2.e_corr - oomp2.e_corr) < abs(0.1 * mp2.e_corr)

uhf = hf.UHF(mol.Molecule(atoms='H 0 0 0; Be 0 0 1.00', basis='sto3g', spin=1)).run()
mp2 = mp.MP2(uhf).run()
oomp2 = mp.OOMP2(uhf).run()
assert abs(mp2.e_corr - oomp2.e_corr) < abs(0.1 * mp2.e_corr)
