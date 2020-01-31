import numpy as np

from auxgf import hf, mol, cc


mol1 = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto3g')

rhf1 = hf.RHF(mol1)
uhf1 = hf.UHF(mol1)

cc2r1 = cc.CCSD(rhf1)
cc2u1 = cc.CCSD(uhf1)

assert np.allclose(cc2r1.e_tot, [cc2u1.e_tot])
