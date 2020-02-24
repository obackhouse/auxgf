''' Restricted open-shell Hartree-Fock class.
'''

import numpy as np
from pyscf import scf, lib

from auxgf import util
from auxgf.util import log
from auxgf.mol import mol
from auxgf.hf import hf, uhf

class ROHF(hf.HF):
    ''' Restricted open-shell Hartree-Fock class.

    Parameters
    ----------
    mol : Molecule
        object defining the molecule

    See auxgf.hf.hf.HF for additional keyword arguments

    Attributes
    ----------
    chempot : float
        chemical potential
    spin_square : float
        average value of total spin operator <S^2>

    See auxgf.hf.hf.HF for additional attributes
    '''

    def __init__(self, mol, **kwargs):
        raise NotImplementedError
        #TODO Fock build

        kwargs['method'] = scf.ROHF
        super().__init__(mol, **kwargs)

    spin_square = property(uhf.UHF.spin_square)
    chempot = property(uhf.UHF.chempot)
    get_fock = staticmethod(uhf.UHF.get_fock)
    energy_1body = staticmethod(uhf.UHF.energy_1body)

    @classmethod
    def from_pyscf(cls, hf):
        ''' Builds the ROHF object from a pyscf.scf.hf.ROHF object.

        Parameters
        ----------
        hf : pyscf.scf.hf.ROHF
            Hartree-Fock object

        Returns
        -------
        hf : ROHF
            Hartree-Fock object
        '''

        _hf = ROHF(mol.Molecule.from_pyscf(hf.mol))

        _hf._pyscf = hf
        _hf._eri_ao = util.restore(1, _hf.mol._pyscf.intor('int2e'), hf.mol.nao)
        
        return _hf

