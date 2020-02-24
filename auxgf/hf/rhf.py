''' Restricted Hartree-Fock class.
'''

import numpy as np
from pyscf import scf, lib

from auxgf import util
from auxgf.util import log
from auxgf.mol import mol
from auxgf.hf import hf

class RHF(hf.HF):
    ''' Restricted Hartree-Fock class.

    Parameters
    ----------
    mol : Molecule
        object defining the molecule

    See auxgf.hf.hf.HF for additional keyword arguments

    Attributes
    ----------
    chempot : float
        chemical potential

    See auxgf.hf.hf.HF for additional attributes

    Raises
    ------
    RuntimeError
        input `mol` is not closed-shell
    '''

    def __init__(self, mol, **kwargs):
        if mol.nelec % 2:
            raise RuntimeError('hf.RHF requires a closed-shell system.')

        kwargs['method'] = scf.RHF
        super().__init__(mol, **kwargs)

    @property
    def chempot(self):
        homo = util.amax(self.e[self.occ > 0])
        lumo = util.amin(self.e[self.occ == 0])
        return 0.5 * (homo + lumo)

    @staticmethod
    def get_fock(h1e, rdm1, eri):
        ''' Builds the Fock matrix according to the RHF functional.

        Parameters
        ----------
        h1e : (n,n) array
            one-electron core Hamiltonian
        rdm1 : (n,n) array
            one-body reduced density matrix
        eri : (n,n,n,n) array
            electronic repulsion integrals

        Returns
        -------
        fock : (n,n) array
            Fock matrix
        '''

        eri = np.asarray(eri)
        eri = util.restore(8, eri, h1e.shape[0])

        j, k = scf.hf._vhf.incore(eri, rdm1, hermi=1)

        fock = h1e + j - 0.5 * k

        return fock

    @staticmethod
    def energy_1body(h1e, rdm1, fock=None, eri=None):
        ''' Calculates the energy according to the RHF density. Basis
            of input arrays should match.

        Parameters
        ----------
        h1e : (n,n) array
            one-electron core Hamiltonian
        rdm1 : (n,n) array
            one-body reduced density matrix
        fock : (n,n) array, optional
            Fock matrix
        eri : (n,n,n,n) array, optional
            electronic repulsion integrals

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
            fock = RHF.get_fock(h1e, rdm1, eri)
        elif fock is None and eri is None:
            raise ValueError('auxgf.hf.rhf.energy_1body requires either '
                             'fock or eri as keyword arguments.')

        e_1body = 0.5 * np.sum(rdm1 * (h1e + fock))

        return e_1body

    @classmethod
    def from_pyscf(cls, hf):
        ''' Builds the RHF object from a pyscf.scf.hf.RHF object.

        Parameters
        ----------
        hf : pyscf.scf.hf.RHF
            Hartree-Fock object

        Returns
        -------
        hf : RHF
            Hartree-Fock object
        '''

        _hf = RHF(mol.Molecule.from_pyscf(hf.mol))

        _hf._pyscf = hf
        _hf._eri_ao = util.restore(1, _hf.mol._pyscf.intor('int2e'), hf.mol.nao)
        
        return _hf
