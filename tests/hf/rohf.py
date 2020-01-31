import numpy as np
from pyscf import scf

from auxgf import hf, mol


mol1 = mol.Molecule(atoms='H 0 0 0; Be 0 0 1', basis='cc-pvdz', spin=1)

rohf1 = scf.ROHF(mol1._pyscf).run()
rohf2 = hf.ROHF(mol1)
rohf3 = hf.ROHF(mol1, disable_omp=False)
rohf4 = hf.ROHF(mol1, check_stability=False)

assert np.allclose(rohf1.e_tot, [rohf2.e_tot, rohf3.e_tot, rohf4.e_tot])
assert np.allclose(rohf1.get_fock(), rohf2.get_fock(rohf2.h1e_ao, rohf2.rdm1_ao, rohf2.eri_ao))
