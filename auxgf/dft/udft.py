''' Unrestricted density-functional theory class.
'''

import numpy as np
from pyscf import lib
from pyscf import dft as _dft
from pyscf import __version__ as pyscf_version

from auxgf import util
from auxgf.util import log
from auxgf.mol import mol
from auxgf.dft import dft
from auxgf.hf import uhf

class UDFT(uhf.UHF):
    ''' Unrestricted density-functional theory class.

    Parameters
    ----------
    mol : Molecule
        object defining the molecule

    See auxgf.dft.dft.DFT for additional keyword arguments

    Attributes
    ----------
    chempot : float
        chemical potential
    spin_square : float
        average value of total spin operator <S^2>

    See auxgf.dft.dft.DFT for additional attributes
    '''

    def __init__(self, mol, **kwargs):
        self.disable_omp = False
        self.check_stability = False
        self.stability_cycles = 10

        self.mol = mol
        self._pyscf = _dft.UKS(self.mol._pyscf, **kwargs)

    @classmethod
    def from_pyscf(cls, ks):
        ''' Builds the UDFT object from a pyscf.dft.dft.UKS object.

        Parameters
        ----------
        ks : pyscf.dft.dft.UKS
            Unrestricted Kohn-Sham object

        Returns
        -------
        ks : UDFT
            Unrestricted density-functional theory object
        '''

        _ks = UDFT(mol.Molecule.from_pyscf(ks.mol))

        _ks._pyscf = ks
        _ks._eri_ao = util.restore(1, _ks.mol._pyscf.intor('int2e'), ks.mol.nao)

        return _ks
