import numpy as np

from auxgf import mol, hf, mp


rhf = hf.RHF(mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='sto3g'))
mp2 = mp.MP2(rhf)
oomp2 = mp.OOMP2(rhf)
assert abs(mp2.e_corr - oomp2.e_corr) < abs(0.1 * mp2.e_corr)

uhf = hf.UHF(mol.Molecule(atoms='H 0 0 0; Be 0 0 1.00', basis='sto3g', spin=1))
mp2 = mp.MP2(uhf)
oomp2 = mp.OOMP2(uhf)
assert abs(mp2.e_corr - oomp2.e_corr) < abs(0.1 * mp2.e_corr)
