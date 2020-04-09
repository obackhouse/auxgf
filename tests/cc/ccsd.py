import numpy as np

from auxgf import hf, mol, cc


mol1 = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto3g')

rhf1 = hf.RHF(mol1).run() 
uhf1 = hf.UHF(mol1).run()

cc2r1 = cc.CCSD(rhf1).run()
cc2u1 = cc.CCSD(uhf1).run()

assert np.allclose(cc2r1.e_tot, [cc2u1.e_tot])
