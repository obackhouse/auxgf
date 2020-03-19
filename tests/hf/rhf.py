import numpy as np
from pyscf import scf

from auxgf import hf, mol


mol1 = mol.Molecule(atoms='H 0 0 0; Li 0 0 1', basis='cc-pvdz')

rhf1 = scf.RHF(mol1._pyscf).run()
rhf2 = hf.RHF(mol1)
rhf3 = hf.RHF(mol1, disable_omp=False)
rhf4 = hf.RHF(mol1, check_stability=False)

assert np.allclose(rhf1.e_tot, [rhf2.e_tot, rhf3.e_tot, rhf4.e_tot])
assert np.allclose(rhf1.get_fock(), rhf2.get_fock(rhf2.rdm1_ao))
