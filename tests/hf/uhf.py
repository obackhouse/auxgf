import numpy as np
from pyscf import scf

from auxgf import hf, mol
from auxgf.util import mpi


mol1 = mol.Molecule(atoms='H 0 0 0; Be 0 0 1', basis='cc-pvdz', spin=1)

uhf1 = scf.UHF(mol1._pyscf).run()
uhf2 = hf.UHF(mol1).run()
uhf3 = hf.UHF(mol1, disable_omp=False).run()
uhf4 = hf.UHF(mol1, check_stability=False).run()
uhf5 = hf.UHF(mol1, with_df=True).run()

assert np.allclose(uhf1.e_tot, [uhf2.e_tot, uhf3.e_tot, uhf4.e_tot])
assert np.allclose(uhf1.get_fock(dm=uhf2.rdm1_ao), uhf2.get_fock(uhf2.rdm1_ao))
assert uhf5.eri_ao.ndim == 3
