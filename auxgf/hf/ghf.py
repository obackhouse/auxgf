''' Generalised Hartree-Fock class.
'''

import numpy as np
from pyscf import scf, lib

from auxgf import util
from auxgf.util import log
from auxgf.mol import mol
from auxgf.hf import hf, uhf

class GHF(hf.HF):
    ''' Generalised Hartree-Fock class.

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
        #TODO ERI conversion nao inconsistency

        kwargs['method'] = scf.GHF
        super().__init__(mol, **kwargs)

    spin_square = uhf.UHF.spin_square
    chempot = uhf.UHF.chempot
    get_fock = uhf.UHF.get_fock
    energy_1body = uhf.UHF.energy_1body

    @classmethod
    def from_pyscf(cls, hf):
        ''' Builds the GHF object from a pyscf.scf.hf.GHF object.

        Parameters
        ----------
        hf : pyscf.scf.hf.GHF
            Hartree-Fock object

        Returns
        -------
        hf : GHF
            Hartree-Fock object
        '''

        _hf = GHF(mol.Molecule.from_pyscf(hf.mol))

        _hf._pyscf = hf
        _hf._eri_ao = util.restore(1, _hf.mol._pyscf.intor('int2e'), hf.mol.nao)
        
        return _hf

