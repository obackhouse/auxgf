import numpy as np
from pyscf import scf

from auxgf import hf, mol


mol1 = mol.Molecule(atoms='H 0 0 0; Be 0 0 1', basis='cc-pvdz', spin=1)

ghf1 = scf.GHF(mol1._pyscf).run()
ghf2 = hf.GHF(mol1)
ghf3 = hf.GHF(mol1, disable_omp=False)
ghf4 = hf.GHF(mol1, check_stability=False)

assert np.allclose(ghf1.e_tot, [ghf2.e_tot, ghf3.e_tot, ghf4.e_tot])
