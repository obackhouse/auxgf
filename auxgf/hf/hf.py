''' Base Hartree-Fock class and routines.
'''

import numpy as np
from pyscf import scf, lib
from pyscf.scf.stability import rhf_stability, uhf_stability

from auxgf import util
from auxgf.util import log, types, mpi


class HF:
    ''' Base class for Hartree-Fock SCF.

    Parameters
    ----------
    mol : Molecule
        object defining the molecule
    method : str, optional
        HF type {'rhf', 'uhf', 'rohf', 'ghf'} (default 'rhf')
    disable_omp : bool, optional
        disable OpenMP parallelism (default True)
    check_stability : bool, optional
        perform stability check on solution (default True)
    stability_cycles : int, optional
        number of stability checks to perform if `check_stability`
        (default 10)
    
    See pyscf.scf.hf.HF for additional keyword arguments

    Attributes
    ----------
    nao : int
        number of atomic orbitals
    nocc : int
        number of occupied orbitals
    nvir : int
        number of virtual orbitals
    nelec : int
        number of electrons
    nalph : int
        number of alpha electrons
    nbeta : int
        number of beta electrons
    occ : ndarray
        occupancy of molecular orbitals
    c : ndarray
        molecular orbital coefficients
    e : ndarray
        molecular orbital energies
    e_elec : float
        electronic energy
    e_nuc : float
        nuclear energy
    e_tot : float
        total energy
    h1e_ao : ndarray
        core Hamiltonian in atomic orbital basis
    h1e_mo : ndarray
        core Hamiltonian in molecular orbital basis
    fock_ao : ndarray
        Fock matrix in atomic orbital basis
    fock_mo : ndarray 
        Fock matrix in molecular orbital basis
    ovlp_ao : ndarray
        overlap matrix in atomic orbital basis
    ovlp_mo : ndarray
        overlap matrix in molecular orbital basis
    rdm1_ao : ndarray
        density matrix in atomic orbital basis
    rdm1_mo : ndarray
        density matrix in molecular orbital basis
    eri_ao : ndarray
        electronic repulsion integrals in atomic orbital basis
    eri_mo : ndarray
        electronic repulsion integrals in molecular orbital basis

    Methods
    -------
    run(**kwargs)
        runs the calculation, see class parameters for arguments.
    '''

    def __init__(self, mol, **kwargs):
        self.mol = mol

        method = kwargs.pop('method', None)
        if method is None:
            method = scf.UHF if self.nelec % 2 else scf.RHF

        self.disable_omp = kwargs.pop('disable_omp', True)
        self.check_stability = kwargs.pop('check_stability', True)
        self.stability_cycles = kwargs.pop('stability_cycles', 10)
        self.with_df = kwargs.pop('with_df', False)
        self.auxbasis = kwargs.pop('auxbasis', None)

        self._pyscf = method(self.mol._pyscf, **kwargs)

    def run(self, **kwargs):
        if self.disable_omp:
            with lib.with_omp_threads(1):
                self._run(**kwargs)
        else:
            self._run(**kwargs)

        if not self._pyscf.converged:
            if mpi.rank:
                log.warn('%s did not converged.' % self.__class__.__name__)

        return self

    def _run(self, **kwargs):
        if self.with_df:
            self._pyscf = self._pyscf.density_fit()
            self._pyscf.with_df.auxbasis = self.auxbasis

        self._pyscf.run(**kwargs)

        if self.check_stability:
            for niter in range(1, self.stability_cycles+1):
                stability = self.stability()

                if isinstance(stability, tuple):
                    internal, external = stability
                else:
                    internal = stability

                if np.allclose(internal, self._pyscf.mo_coeff):
                    if niter == self.stability_cycles:
                        if mpi.rank:
                            log.warn('Internal stability in HF not resolved.')
                    break
                else:
                    rdm1 = self._pyscf.make_rdm1(internal, self._pyscf.mo_occ)

                self._pyscf.scf(dm0=rdm1)

        if not self.with_df:
            self._eri_ao = util.restore(1, self.mol._pyscf.intor('int2e'), 
                                        self.nao)
        else:
            self._eri_ao = lib.unpack_tril(self._pyscf.with_df._cderi)

    def stability(self, **kwargs):
        if self.nelec % 2:
            return uhf_stability(self._pyscf, **kwargs)
        else:
            return rhf_stability(self._pyscf, **kwargs)
    
    @property
    def nao(self):
        return self.occ.shape[-1]

    @property
    def nocc(self):
        return np.sum(self.occ > 0, axis=-1)

    @property
    def nvir(self):
        return np.sum(self.occ == 0, axis=-1)

    @property
    def nelec(self):
        return self.mol.nelec

    @property
    def nalph(self):
        return self.mol.nalph

    @property
    def nbeta(self):
        return self.mol.nbeta

    @property
    def occ(self):
        return np.stack(self._pyscf.mo_occ).astype(types.int32)

    @property
    def c(self):
        return np.stack(self._pyscf.mo_coeff)

    @property
    def e(self):
        return np.stack(self._pyscf.mo_energy)

    @property
    def e_elec(self):
        return self._pyscf.energy_elec()[0]

    @property
    def e_nuc(self):
        return self._pyscf.energy_nuc()

    @property
    def e_tot(self):
        return self._pyscf.e_tot

    @property
    def h1e_ao(self):
        return self._pyscf.get_hcore()

    @property
    def h1e_mo(self):
        return util.ao2mo(self.h1e_ao, self.c, self.c)

    @property
    def fock_ao(self):
        return self._pyscf.get_fock()
    
    @property
    def fock_mo(self):
        return util.ao2mo(self.fock_ao, self.c, self.c)

    @property
    def ovlp_ao(self):
        return self._pyscf.get_ovlp()

    @property
    def ovlp_mo(self):
        return np.eye(self.nao)

    @property
    def rdm1_ao(self):
        return self._pyscf.make_rdm1()

    @property
    def rdm1_mo(self):
        c = self.c
        s = self.ovlp_ao
        d = self.rdm1_ao
        return util.einsum('...ba,cb,...cd,de,...ef->...af', c, s, d, s, c)

    @property
    def eri_ao(self):
        return self._eri_ao

    @property
    def eri_mo(self):
        if not self.with_df:
            return util.ao2mo(self.eri_ao, self.c, self.c, self.c, self.c)
        else:
            return util.ao2mo(self.eri_ao, self.c, self.c)

    def get_eri_mo(self, masks):
        coeffs = [self.c[:,mask] for mask in masks]
        return util.ao2mo(self.eri_ao, *coeffs)
