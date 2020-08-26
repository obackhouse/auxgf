''' Restricted Hartree-Fock class.
'''

import numpy as np
from pyscf import scf, lib
from pyscf.scf.stability import rhf_stability

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

    def get_fock(self, rdm1, basis='ao'):
        ''' Builds the Fock matrix according to the RHF functional.

        Parameters
        ----------
        rdm1 : (n,n) ndarray
            one-body reduced density matrix
        basis : str, optional
            input basis of `rdm1`, and output basis of `fock`, 
            default 'ao'

        Returns
        -------
        fock : (n,n) array
            Fock matrix
        '''

        c = self.c if basis == 'mo' else np.eye(self.nao)

        dm = util.einsum('ij,pi,qj->pq', rdm1, c, c)
        fock = self._pyscf.get_fock(dm=dm)
        fock = util.einsum('pq,pi,qj->ij', fock, c, c)

        return fock

    @staticmethod
    def energy_1body(h1e, rdm1, fock):
        ''' Calculates the energy according to the RHF density. Basis
            of input arrays should match.

        Parameters
        ----------
        h1e : (n,n) array
            one-electron core Hamiltonian
        rdm1 : (n,n) array
            one-body reduced density matrix
        fock : (n,n) array
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

        if not getattr(hf, 'with_df', False):
            _hf._eri_ao = util.restore(1, _hf.mol._pyscf.intor('int2e'), hf.mol.nao)
        else:
            if hf.with_df._cderi is None:
                hf.with_df.run()

            _hf._eri_ao = lib.unpack_tril(hf.with_df._cderi)
        
        return _hf

    def stability(self, **kwargs):
        return rhf_stability(self._pyscf, **kwargs)

