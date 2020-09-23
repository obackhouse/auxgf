''' Class to perform DF-UAGF2(None,0) with efficient MPI parallel
    algorithms. Most of this code will remain self-contained for
    the sake of optimisation.
'''

import numpy as np
from pyscf import lib
from pyscf.ao2mo import _ao2mo
import functools
import ctypes

from auxgf import util, aux
from auxgf.util import types, log, mpi
from auxgf.lib import agf2 as libagf2
from auxgf.agf2.fock import fock_loop_uhf
from auxgf.mpi.agf2 import optragf2


_set_options = optragf2._set_options


util.reshape_internal = lambda x, s1, swap, s2 : \
                           x.reshape(s1).swapaxes(*swap).reshape(s2)

_fdrv = functools.partial(_ao2mo.libao2mo.AO2MOnr_e2_drv, 
                          _ao2mo.libao2mo.AO2MOtranse2_nr_s2,
                          _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2)

to_ptr = lambda m : m.ctypes.data_as(ctypes.c_void_p)


class OptUAGF2(util.AuxMethod):
    ''' Unrestricted auxiliary GF2 method for (None,0) and DF integrals.

    Parameters
    ----------
    uhf : UHF
        Hartree-Fock object
    dm0 : (2,n,n) ndarray, optional
        initial density matrix, if None, use rhf.rdm1_mo, default
        None
    verbose : bool, optional
        if True, print output log, default True
    maxiter : int, optional
        maximum number of RAGF2 iterations, default 50
    etol : float, optional
        maximum difference in subsequent energies at convergence,
        default 1e-6
    wtol : float, optional
        minimum pole weight to be considered zero, default 1e-12
    damping : float, optional
        self-energy damping factor via
            S(i) = damping * S(i) + (1-damping) * S(i-1),
        default 0.5
    delay_damping : int, optional
        skip this number of iterations before starting to damp,
        default 0
    dtol : float, optional
        maximum difference in density matrices at convergence in the
        Fock loop, default 1e-8
    diis_space : int, optional  
        size of DIIS space for inner Fock loop, default 8
    fock_maxiter : int, optional
        maximum number of inner Fock loop iterations, default 50
    fock_maxruns : int, optional
        maximum number of outer Fock loop iterations, default 20
    maxblk : float, optional
        maximum block of density fitted integrals
    ss_factor : float, optional
        same spin factor for auxiliary build, default 1.0
    os_factor : float, optional
        opposite spin factor for auxiliary build, default 1.0

    Attributes
    ----------
    hf : UHF
        Hartree-Fock object
    nmom : tuple of int
        returns (None, 0)
    verbose : bool
        see parameters
    options : dict
        dictionary of options
    rdm1 : (2,n,n) ndarray
        one-particle reduced density matrix, projected into the
        physical basis
    converged : bool
        whether the method has converged
    iteration : int
        the current/final iteration reached
    se : tuple of Aux
        auxiliary representation of the self-energy for alpha, beta
    e_1body : float
        one-body energy
    e_2body : float
        two-body energy
    e_tot : float
        total energy
    e_mp2 : float
        MP2 energy (equivalent to `e2b` at `iteration=0`)
    e_corr : float
        correlation energy i.e. `e_tot` - `hf.e_tot`

    Methods
    -------
    setup(rhf)
        constructs the object using the parameters provided to 
        `__init__` and performs the initial MP2 iteration
    get_fock(rdm1=None)
        returns the Fock matrix resulting from the current, or
        provided, density
    run()
        runs the method
    '''

    def __init__(self, uhf, **kwargs):
        super().__init__(uhf, **kwargs)

        self.options = _set_options(self.options, **kwargs)

        if mpi.mpi is None:
            log.warn('No MPI4Py installation detected, OptUAGF2 will therefore run in serial.')

        self.setup()


    @util.record_time('setup')
    def setup(self):
        super().setup()

        self.gf = (self.se[0].new(self.hf.e[0], np.eye(self.hf.nao)),
                   self.se[1].new(self.hf.e[1], np.eye(self.hf.nao)))

        if self.eri.ndim == 4:
            self.eri = np.stack([lib.pack_tril(x) for x in self.eri], axis=0)

        log.title('Options', self.verbose)
        log.options(self.options, self.verbose)
        log.title('Input', self.verbose)
        log.molecule(self.hf.mol, self.verbose)
        log.write('Basis = %s\n' % self.hf.mol.basis, self.verbose)
        log.write('E(nuc) = %.12f\n' % self.hf.e_nuc, self.verbose)
        log.write('E(hf)  = %.12f\n' % self.hf.e_tot, self.verbose)
        log.write('nao = %d\n' % self.hf.nao, self.verbose)
        log.write('nmom = (%s, %s)\n' % self.nmom, self.verbose)

        self.run_mp2()


    #@staticmethod
    #def build_x(ixq, qja, nphys, nocc, nvir):
    #    ''' Builds the X array, entirely equivalent to the zeroth-
    #        order moment matrix of the self-energy.
    #    '''

    #    nocca, noccb = nocc
    #    nvira, nvirb = nvir

    #    x = np.zeros((nphys, nphys), dtype=types.float64)
    #    buf1 = np.zeros((nphys, nocca*nvira), dtype=types.float64)
    #    buf2 = np.zeros((nocca*nphys, nvira), dtype=types.float64)
    #    buf3 = np.zeros((nphys, noccb*nvirb), dtype=types.float64)
    #    
    #    for i in range(mpi.rank, nocca, mpi.size):
    #        xja_aa = np.dot(ixq[0][i*nphys:(i+1)*nphys], qja[0], out=buf1)
    #        xia_aa = np.dot(ixq[0], qja[0][:,i*nvira:(i+1)*nvira], out=buf2)
    #        xia_aa = util.reshape_internal(xia_aa, (nocca, nphys, nvira), (0,1), (nphys, nocca*nvira))
    #        xja_ab = np.dot(ixq[0][i*nphys:(i+1)*nphys], qja[1], out=buf3)

    #        x = util.dgemm(xja_aa, xja_aa.T, alpha=1, beta=1, c=x)
    #        x = util.dgemm(xja_aa, xia_aa.T, alpha=-1, beta=1, c=x)
    #        x = util.dgemm(xja_ab, xja_ab.T, alpha=1, beta=1, c=x)

    #    x = mpi.reduce(x)

    #    return x


    #@staticmethod
    #def build_m(gf_occ, gf_vir, ixq, qja, b_inv):
    #    ''' Builds the M array.
    #    '''

    #    nphys = gf_occ[0].nphys
    #    nocca, noccb = (gf_occ[0].naux, gf_occ[1].naux)
    #    nvira, nvirb = (gf_vir[0].naux, gf_vir[1].naux)

    #    m = np.zeros((nphys, nphys), dtype=types.float64)

    #    eo = (gf_occ[0].e, gf_occ[1].e)
    #    ev = (gf_vir[0].e, gf_vir[1].e)
    #    indices = mpi.tril_indices_rows(nocca)
    #    a_factor = np.sqrt(1.0)
    #    b_factor = np.sqrt(1.0)

    #    for i in indices[mpi.rank]:
    #        xq_a = ixq[0][i*nphys:(i+1)*nphys]
    #        qa_a = qja[0][:,i*nvira:(i+1)*nvira]

    #        xja_aa = np.dot(ixq[0][:i*nphys], qa_a)
    #        xja_aa = util.reshape_internal(xja_aa, (i, nphys, nvira), (0,1), (nphys, i*nvira))
    #        xia_aa = np.dot(xq_a, qja[0][:,:i*nvira]).reshape((nphys, -1))
    #        xja_ab = np.dot(xq_a, qja[1]).reshape((nphys, -1))

    #        ea = eo[0][i] + util.dirsum('i,a->ia', eo[0][:i], -ev[0]).ravel()
    #        eb = eo[0][i] + util.dirsum('i,a->ia', eo[1], -ev[1]).ravel()

    #        va = a_factor * (xia_aa - xja_aa)
    #        vb = b_factor * xja_ab

    #        qa = np.dot(b_inv.T, va)
    #        qb = np.dot(b_inv.T, vb)

    #        m = util.dgemm(qa * ea[None], qa.T, c=m, beta=1)
    #        m = util.dgemm(qb * eb[None], qb.T, c=m, beta=1)

    #    m = mpi.reduce(m)

    #    return m


    @staticmethod
    def build_part(gf_occ, gf_vir, eri, sym_in='s2'):
        ''' Builds the truncated occupied (or virtual) self-energy.

        Parameters
        ----------
        gf_occ : Aux
            Occupied (or virtual) Green's function for alpha, beta spins
        gf_vir : Aux
            Virtual (or occupied) Green's function for alpha, beta spins
        eri : ndarray
            Cholesky-decomposed DF ERI tensor for alpha, beta spins
        sym_in : str, optional
            Symmetry of `eri`, default 's2'

        Returns
        -------
        se : Aux
            Occupied (or virtual) truncated self-energy for alpha, beta
            spins
        '''

        syms = dict(sym_in=sym_in, sym_out='s1')
        nphys = gf_occ[0].nphys
        nocc = (gf_occ[0].naux, gf_occ[1].naux)
        nvir = (gf_vir[0].naux, gf_vir[1].naux)

        ixq_a = optragf2.OptRAGF2.ao2mo(eri[0], gf_occ[0].v, np.eye(nphys), **syms).T
        qja_a = optragf2.OptRAGF2.ao2mo(eri[0], gf_occ[0].v, gf_vir[0].v, **syms)
        ixq_b = optragf2.OptRAGF2.ao2mo(eri[1], gf_occ[1].v, np.eye(nphys), **syms).T
        qja_b = optragf2.OptRAGF2.ao2mo(eri[1], gf_occ[1].v, gf_vir[1].v, **syms)

        ixq = (ixq_a, ixq_b)
        qja = (qja_a, qja_b)

        def _build_part(s=slice(None)):
            vv = np.zeros((nphys, nphys), dtype=types.float64)
            vev = np.zeros((nphys, nphys), dtype=types.float64)

            if libagf2._libagf2 is not None:
                parts = mpi.split_int(nocc[s][0])
                istart = sum(parts[:mpi.rank])
                iend = sum(parts[:(mpi.rank+1)])

                vv, vev = libagf2.build_part_loop_uhf(ixq[s], qja[s], gf_occ[s], gf_vir[s], istart, iend, vv=vv, vev=vev)

            else:
                nocca, noccb = nocc[s]
                nvira, nvirb = nvir[s]
                ixq_a, ixq_b = ixq[s]
                qja_a, qja_b = qja[s]
                gf_occ_a, gf_occ_b = gf_occ[s]
                gf_vir_a, gf_vir_b = gf_vir[s]

                buf1 = np.zeros((nphys, nocca*nvira), dtype=types.float64)
                buf2 = np.zeros((nocca*nphys, nvira), dtype=types.float64)
                buf3 = np.zeros((nphys, noccb*nvirb), dtype=types.float64)

                for i in range(mpi.rank, nocca, mpi.size):
                    xja_aa = np.dot(ixq_a[i*nphys:(i+1)*nphys], qja_a, out=buf1)
                    xia_aa = np.dot(ixq_a, qja_a[:,i*nvira:(i+1)*nvira], out=buf2)
                    xia_aa = util.reshape_internal(xia_aa, (nocca, nphys, nvira), (0,1), (nphys, nocca*nvira))
                    xja_ab = np.dot(ixq_a[i*nphys:(i+1)*nphys], qja_b, out=buf3)

                    eja_aa = util.outer_sum([gf_occ_a.e[i] + gf_occ_a.e, -gf_vir_a.e]).flatten() 
                    eja_ab = util.outer_sum([gf_occ_a.e[i] + gf_occ_b.e, -gf_vir_b.e]).flatten()

                    vv = util.dgemm(xja_aa, xja_aa.T, alpha=1, beta=1, c=vv)
                    vv = util.dgemm(xja_aa, xia_aa.T, alpha=-1, beta=1, c=vv)
                    vv = util.dgemm(xja_ab, xja_ab.T, alpha=1, beta=1, c=vv)

                    vev = util.dgemm(xja_aa * eja_aa[None], xja_aa.T, alpha=1, beta=1, c=vev)
                    vev = util.dgemm(xja_aa * eja_aa[None], xia_aa.T, alpha=-1, beta=1, c=vev)
                    vev = util.dgemm(xja_ab * eja_ab[None], xja_ab.T, alpha=1, beta=1, c=vev)

            vv = mpi.reduce(vv)
            vev = mpi.reduce(vev)

            b = np.linalg.cholesky(vv).T
            b_inv = np.linalg.inv(b)

            m = np.dot(np.dot(b_inv.T, vev), b_inv)

            e, c = util.eigh(m)
            c = np.dot(b.T, c[:nphys])

            se = gf_occ[s][0].new(e, c)
            
            return se

        se_a = _build_part(s=slice(None, None, 1))
        se_b = _build_part(s=slice(None, None, -1))

        return se_a, se_b


    @util.record_time('build')
    def build(self):
        self.solve_dyson()

        gf_occ = (self.gf[0].as_occupied(), self.gf[1].as_occupied())
        gf_vir = (self.gf[0].as_virtual(), self.gf[1].as_virtual())

        se_occ = self.build_part(gf_occ, gf_vir, self.eri)
        se_vir = self.build_part(gf_vir, gf_occ, self.eri)

        self.se = (se_occ[0] + se_vir[0], se_occ[1] + se_vir[1])


    @util.record_time('fock')
    def fock_loop(self):
        fock_opts = self.options['_fock_loop']
        fock_opts['fock_func'] = self.get_fock

        se, rdm1, converged = fock_loop_uhf(self.se, self.hf, self.rdm1, **fock_opts)

        self.solve_dyson()

        if converged:
            log.write('Fock loop converged.\n', self.verbose)
            log.write('Chemical potential (alpha) = %.6f\n' % self.chempot[0], self.verbose)
            log.write('Chemical potential (beta)  = %.6f\n' % self.chempot[1], self.verbose)
        else:
            log.write('Fock loop did not converge.\n', self.verbose)
            log.write('Chemical potential (alpha) = %.6f\n' % self.chempot[0], self.verbose)
            log.write('Chemical potential (beta)  = %.6f\n' % self.chempot[1], self.verbose)


    @util.record_time('energy')
    @util.record_energy('mp2')
    def energy_mp2(self):
        emp2_a = aux.energy.energy_mp2_aux(self.hf.e[0], self.se[0])
        emp2_b = aux.energy.energy_mp2_aux(self.hf.e[1], self.se[1])

        emp2 = emp2_a + emp2_b
        emp2 /= 2

        log.write('E(mp2) = %.12f\n' % emp2, self.verbose)

        return emp2

    @util.record_time('energy')
    @util.record_energy('1b')
    def energy_1body(self):
        e1b = self.hf.energy_1body(self.h1e, self.rdm1, self.get_fock())
        e1b += self.hf.mol.e_nuc

        log.write('E(1b)  = %.12f\n' % e1b, self.verbose)

        return e1b

    @util.record_time('energy')
    @util.record_energy('2b')
    def energy_2body(self):
        self.solve_dyson()

        e2b = aux.energy_2body_aux(self.gf, self.se)

        log.write('E(2b)  = %.12f\n' % e2b, self.verbose)

        return e2b

    @util.record_time('energy')
    @util.record_energy('tot')
    def energy_tot(self):
        etot = self.e_1body + self.e_2body

        log.write('E(tot) = %.12f\n' % etot, self.verbose)

        return etot

    def energy(self):
        self.energy_1body()
        self.energy_2body()
        self.energy_tot()


    def get_fock(self, rdm1=None):
        ''' Returns the Fock matrix resulting from the current, or
            provided, density.
        '''

        if rdm1 is None:
            rdm1 = self.rdm1

        nphys = self.nphys
        naux = self.eri.shape[1]
        maxblk = self.options['maxblk']

        def _get_jk(_rdm1, _eri):
            j = np.zeros((nphys*(nphys+1)//2), dtype=types.float64)
            k = np.zeros((nphys, nphys), dtype=types.float64)

            _rdm1_tril = lib.pack_tril(_rdm1 + np.tril(_rdm1, k=-1))
            c_int = ctypes.c_int
            args = (c_int(nphys), (c_int*4)(0, nphys, 0, nphys), lib.c_null_ptr(), c_int(0))
            buf = np.empty((2, maxblk, nphys, nphys))

            for s in mpi.get_blocks(naux, maxblk, all_ranks=False):
                eri1 = _eri[s]
                naux_block = eri1.shape[0]

                rho = np.dot(eri1, _rdm1_tril)
                j += np.dot(rho, eri1)

                buf1 = buf[0,:naux_block]
                _fdrv(to_ptr(buf1), to_ptr(eri1), to_ptr(_rdm1), c_int(naux_block), *args)

                buf2 = lib.unpack_tril(eri1, out=buf[1])
                k = util.dgemm(buf1.reshape((-1, nphys)).T, buf2.reshape((-1, nphys)), c=k, beta=1)

            j = mpi.reduce(j)
            k = mpi.reduce(k)

            j = lib.unpack_tril(j).reshape(_rdm1.shape)
            k = k.reshape(_rdm1.shape)

            return j, k

        j_a, k_a = _get_jk(rdm1[0], self.eri[0])
        j_b, k_b = _get_jk(rdm1[1], self.eri[1])

        f_a = self.h1e[0] + j_a + j_b - k_a
        f_b = self.h1e[1] + j_b + j_a - k_b

        f = np.stack([f_a, f_b], axis=0)

        return f


    def run_mp2(self):
        log.iteration(0, self.verbose)

        self.build()
        self.energy_mp2()


    def run(self):
        maxiter = self.options['maxiter']
        etol = self.options['etol']
        checkpoint = self.options['checkpoint']

        for self.iteration in range(1, maxiter+1):
            log.iteration(self.iteration, self.verbose)

            self.fock_loop()
            self.build()
            self.energy()

            if self.iteration > 1:
                e_dif = abs(self._energies['tot'][-2] - self._energies['tot'][-1])

                if e_dif < etol and self.converged:
                    break

                self.converged = e_dif < etol

            if checkpoint and mpi.rank == 0:
                np.savetxt('rdm1_alph_chk.dat', self.rdm1[0])
                np.savetxt('rdm1_beta_chk.dat', self.rdm1[1])
                self.se[0].save('se_alph_chk.pickle')
                self.se[1].save('se_beta_chk.pickle')

        if self.converged:
            log.write('\nAuxiliary GF2 converged after %d iterations.\n'
                      % self.iteration, self.verbose)
        else:
            log.write('\nAuxiliary GF2 failed to converge.\n', self.verbose)

        self._timings['total'] = self._timings.get('total', 0.0) + self._timer.total()
        log.title('Timings', self.verbose)
        log.timings(self._timings, self.verbose)

        return self


    @property
    def e_2body(self):
        if self.iteration:
            return self._energies['2b'][-1]
        else:
            return self.e_mp2

    @property
    def e_tot(self):
        if self.iteration:
            return self._energies['tot'][-1]
        else:
            return self.e_hf + self.e_mp2

    @property
    def e_mp2(self):
        return self._energies['mp2'][-1]

    @property
    def nmom(self):
        return (None, 0)
