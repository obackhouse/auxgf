import numpy as np
from pyscf import scf

from auxgf import hf, mol


mol1 = mol.Molecule(atoms='H 0 0 0; Be 0 0 1', basis='cc-pvdz', spin=1)

uhf1 = scf.UHF(mol1._pyscf).run()
uhf2 = hf.UHF(mol1)
uhf3 = hf.UHF(mol1, disable_omp=False)
uhf4 = hf.UHF(mol1, check_stability=False)

assert np.allclose(uhf1.e_tot, [uhf2.e_tot, uhf3.e_tot, uhf4.e_tot])
