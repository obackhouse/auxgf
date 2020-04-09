# Load in molecules and HF calculations from pyscf

import numpy as np
from pyscf import gto, scf
from auxgf import mol, hf
from auxgf.util import Timer

timer = Timer()


# From a pyscf.gto.Mole object:
m_pyscf = gto.M(atom='H 0 0 0; Li 0 0 1.64', basis='cc-pvdz', verbose=False)
m = mol.Molecule.from_pyscf(m_pyscf)

# From a pyscf.scf.RHF object:
rhf_pyscf = scf.RHF(m_pyscf).run()
rhf = hf.RHF.from_pyscf(rhf_pyscf)
assert rhf_pyscf.e_tot == rhf.e_tot


print('time elapsed: %d min %.4f s' % (timer.total() // 60, timer.total() % 60))
