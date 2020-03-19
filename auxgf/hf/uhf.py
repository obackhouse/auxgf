''' Unrestricted Hartree-Fock class.
'''

import numpy as np
from pyscf import scf, lib
from pyscf import __version__ as pyscf_version

from auxgf import util
from auxgf.util import log
from auxgf.mol import mol
from auxgf.hf import hf

class UHF(hf.HF):
    ''' Unrestricted Hartree-Fock class.

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
        kwargs['method'] = scf.UHF
        super().__init__(mol, **kwargs)

    @property
    def spin_square(self):
        return self._pyscf.spin_square()

    @property
    def chempot(self):
        homo_a = util.amax(self.e[0][self.occ[0] > 0])
        lumo_a = util.amin(self.e[0][self.occ[0] == 0]) 
        homo_b = util.amax(self.e[1][self.occ[1] > 0])
        lumo_b = util.amin(self.e[1][self.occ[1] == 0]) 
        chempot_a = 0.5 * (homo_a + lumo_a)
        chempot_b = 0.5 * (homo_b + lumo_b)
        return chempot_a, chempot_b

    def get_fock(self, rdm1, basis='ao'):
        ''' Builds the Fock matrix according to the UHF functional.
            Input arrays may be spin-free or indexed for alpha and
            beta spins.

        Parameters
        ----------
        rdm1 : (n,n) or (2,n,n) array
            one-body reduced density matrix

        Returns
        -------
        fock : (2,n,n) array
            Fock matrix
        '''

        c = self.c if basis == 'mo' else np.eye(self.nao)

        dm = util.einsum('...ij,...pi,...qj->...pq', rdm1, c, c)
        fock = self._pyscf.get_fock(dm=dm)
        fock = util.einsum('...pq,...pi,...qj->...ij', fock, c, c)

        return fock

    @staticmethod
    def energy_1body(h1e, rdm1, fock):
        ''' Calculates the energy according to the UHF density. Basis
            of input arrays should match. Input arrays may be spin-
            free or indexed for alpha and beta spins.

        Parameters
        ----------
        h1e : (n,n) or (2,n,n) array
            one-electron core Hamiltonian
        rdm1 : (n,n) or (2,n,n) array
            one-body reduced density matrix
        fock : (n,n) or (2,n,n) array
            Fock matrix

        Returns
        -------
        e_1body : float
            one-body energy
        '''

        e_1body = 0.5 * np.sum(rdm1 * (h1e + fock))

        return e_1body

    @classmethod
    def from_pyscf(cls, hf):
        ''' Builds the UHF object from a pyscf.scf.hf.UHF object.

        Parameters
        ----------
        hf : pyscf.scf.hf.UHF
            Hartree-Fock object

        Returns
        -------
        hf : UHF
            Hartree-Fock object
        '''

        _hf = UHF(mol.Molecule.from_pyscf(hf.mol))

        _hf._pyscf = hf
        _hf._eri_ao = util.restore(1, _hf.mol._pyscf.intor('int2e'), hf.mol.nao)
        
        return _hf

