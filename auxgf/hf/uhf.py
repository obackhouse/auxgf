''' Unrestricted Hartree-Fock class.
'''

import numpy as np
from pyscf import scf, lib

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

    @staticmethod
    def get_fock(h1e, rdm1, eri):
        ''' Builds the Fock matrix according to the UHF functional.
            Input arrays may be spin-free or indexed for alpha and
            beta spins.

        Parameters
        ----------
        h1e : (n,n) or (2,n,n) array
            one-electron core Hamiltonian
        rdm1 : (n,n) or (2,n,n) array
            one-body reduced density matrix
        eri : (n,n,n,n) or (4,n,n,n,n) or (2,2,n,n,n,n) array
            electronic repulsion integrals

        Returns
        -------
        fock : (2,n,n) array
            Fock matrix
        '''

        n = h1e.shape[-1]

        if h1e.ndim == 2 or h1e.shape == (1, n, n):
            h1e = np.stack((h1e.reshape((n, n)),)*2, axis=0)

        if rdm1.ndim == 2 or rdm1.shape == (1, n, n):
            rdm1 = np.stack((rdm1.reshape((n, n)),)*2, axis=0)

        if eri.ndim == 4 or eri.shape == (1, n, n, n, n) or \
                eri.shape == (1, 1, n, n, n, n):
            eri = np.stack((eri.reshape((n, n, n, n)),)*4, axis=0)
            eri = eri.reshape((2, 2, n, n, n, n))
        elif eri.shape == (4, n, n, n, n):
            eri = eri.reshape((2, 2, n, n, n, n))

        j = util.einsum('aji,abijkl->bkl', rdm1, eri)
        k = util.einsum('aji,ailkj->akl', rdm1, (eri[0,0], eri[1,1]))

        fock = h1e + j - k

        return fock

    @staticmethod
    def energy_1body(h1e, rdm1, fock=None, eri=None):
        ''' Calculates the energy according to the UHF density. Basis
            of input arrays should match. Input arrays may be spin-
            free or indexed for alpha and beta spins.

        Parameters
        ----------
        h1e : (n,n) or (2,n,n) array
            one-electron core Hamiltonian
        rdm1 : (n,n) or (2,n,n) array
            one-body reduced density matrix
        eri : (n,n,n,n) or (4,n,n,n,n) or (2,2,n,n,n,n) array, optional
            electronic repulsion integrals
        fock : (n,n) or (2,n,n) array, optional
            Fock matrix

        Returns
        -------
        e_1body : float
            one-body energy

        Raises
        ------
        ValueError
            neither `fock` nor `eri` were passed as keyword arguments
        '''

        if fock is None and eri is not None:
            fock = UHF.get_fock(h1e, rdm1, eri)
        elif fock is None and eri is None:
            raise ValueError('auxgf.hf.rhf.energy_1body requires either '
                             'fock or eri as keyword arguments.')

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
        _hf = UHF(mol.from_pyscf(hf.mol))

        _hf._pyscf = hf
        _hf._eri_ao = util.restore(1, _hf.mol._pyscf.intor('int2e', hf.mol.nao))
        
        return _hf

