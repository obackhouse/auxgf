''' Restricted density-functional theory class.
'''

import numpy as np
from pyscf import lib
from pyscf import dft as _dft

from auxgf import util
from auxgf.util import log
from auxgf.mol import mol
from auxgf.dft import dft
from auxgf.hf import rhf

class RDFT(rhf.RHF):
    ''' Restricted density-functional theory class.

    Parameters
    ----------
    mol : Molecule
        object defining the molecule

    See auxgf.dft.dft.DFT for additional keyword arguments

    Attributes
    ----------
    chempot : float
        chemical potential

    See auxgf.dft.dft.DFT for additional attributes

    Raises
    ------
    RuntimeError
        input `mol` is not closed-shell
    '''

    def __init__(self, mol, **kwargs):
        if mol.nelec % 2:
            raise RuntimeError('dft.RDFT requires a closed-shell system.')

        self.disable_omp = False
        self.check_stability = False
        self.stability_cycles = 10

        self.mol = mol
        self._pyscf = _dft.RKS(self.mol._pyscf, **kwargs)

    @classmethod
    def from_pyscf(cls, ks):
        ''' Builds the RDFT object from a pyscf.dft.dft.RKS object.

        Parameters
        ----------
        ks : pyscf.dft.dft.RKS
            Restricted Kohn-Sham object

        Returns
        -------
        ks : RDFT
            Restricted density-functional theory object
        '''

        _ks = RDFT(mol.Molecule.from_pyscf(ks.mol))

        _ks._pyscf = ks
        _ks._eri_ao = util.restore(1, _ks.mol._pyscf.intor('int2e'), ks.mol.nao)

        return _ks
