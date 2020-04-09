import numpy as np
from pyscf import dft as _dft

from auxgf import dft, mol


for xc in ['hf', 'pbe', 'b3lyp']:
    mol1 = mol.Molecule(atoms='H 0 0 0; Be 0 0 1', basis='6-31g', spin=1)

    udft1 = _dft.UKS(mol1._pyscf).run(xc=xc)
    udft2 = dft.UDFT(mol1, xc=xc)

    assert np.allclose(udft1.e_tot, udft2.e_tot)
    assert np.allclose(udft1.get_fock(), udft2.get_fock(udft2.rdm1_ao))
