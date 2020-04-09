import numpy as np

from auxgf import hf, mol, mp


mol1 = mol.Molecule(atoms='H 0 0 0; Li 0 0 1', basis='cc-pvdz')

rhf1 = hf.RHF(mol1).run()
uhf1 = hf.UHF(mol1).run()

mp2r1 = mp.MP2(rhf1).run()
mp2u1 = mp.MP2(uhf1).run()

assert np.allclose(mp2r1.e_tot, [mp2u1.e_tot])
