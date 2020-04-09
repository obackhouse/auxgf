import numpy as np
from pyscf import dft as _dft

from auxgf import dft, mol


for xc in ['hf', 'pbe', 'b3lyp']:
    mol1 = mol.Molecule(atoms='H 0 0 0; Li 0 0 1', basis='6-31g')

    rdft1 = _dft.RKS(mol1._pyscf).run(xc=xc)
    rdft2 = dft.RDFT(mol1).run(xc=xc)

    assert np.allclose(rdft1.e_tot, rdft2.e_tot)
    assert np.allclose(rdft1.get_fock(), rdft2.get_fock(rdft2.rdm1_ao))
