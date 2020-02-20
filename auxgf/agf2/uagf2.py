''' Class to perform auxiliary GF2 calculations for unrestricted
    references.
'''

import numpy as np
import functools

from auxgf import util, grids, aux
from auxgf.util import types, log
from auxgf.agf2.fock import fock_loop_uhf


def _set_options(**kwargs):
    options = { 'nmom' : (3,4),
                'dm0' : None,
                'frozen' : 0,
                'verbose' : True,
                'maxiter' : 50,
                'etol' : 1e-6,
                'wtol' : 1e-10,
                'damping' : 0.5,
                'dtol' : 1e-8,
                'diis_space' : 8,
                'fock_maxiter' : 50,
                'fock_maxruns' : 20,
                'ss_factor' : 1.0,
                'os_factor' : 1.0,
                'use_merge' : False,
                'bath_type' : 'power',
                'bath_beta' : 100,
    }

    for key,val in kwargs.items():
        if key not in options.keys():
            raise ValueError('%s argument invalid.' % key)

    options.update(kwargs)

    options['_fock_loop'] = {
        'dtol' : options['dtol'],
        'diis_space' : options['diis_space'],
        'maxiter' : options['fock_maxiter'],
        'maxruns' : options['fock_maxruns'],
        'frozen' : options['frozen'],
        'verbose' : options['verbose'],
    }

    options['_build'] = {
        'wtol' : options['wtol'],
        'ss_factor' : options['ss_factor'],
        'os_factor' : options['os_factor'],
    }

    return options


def _active(uagf2, arr):
    ''' Returns the active space of an n-dimensional array.
    '''

    frozen = uagf2.options['frozen']

    if not frozen:
        return arr

    # Filter out spin indices (could get confused for nao == 2 but
    # such systems should never have frozen orbitals):
    ndim = sum(x == uagf2.hf.nao for x in arr.shape)
    act = (Ellipsis,) + (slice(frozen, None),)*ndim

    return arr[act]


class UAGF2:
    ''' Unrestricted auxiliary GF2 method.

    Parameters
    ----------
    uhf : UHF
        Unrestricted Hartree-Fock object
    nmom : tuple of int, optional
        number of moments to which the truncation is consistent to,
        ordered by (Green's function, self-energy), default (3,4), 
        see auxgf.aux.aux.Aux.compress
    dm0 : (2,n,n) ndarray, optional
        initial density matrix for alpha and beta spins, if None, 
        use rhf.rdm1_mo, default None
    frozen : int, optional
        number of frozen core orbitals, default 0
    verbose : bool, optional
        if True, print output log, default True
    maxiter : int, optional
        maximum number of RAGF2 iterations, default 50
    etol : float, optional
        maximum difference in subsequent energies at convergence,
        default 1e-6
    wtol : float, optional
        minimum pole weight to be considered zero, default 1e-10
    damping : float, optional
        self-energy damping factor via
            S(i) = damping * S(i) + (1-damping) * S(i-1),
        default 0.5
    dtol : float, optional
        maximum difference in density matrices at convergence in the
        Fock loop, default 1e-8
    diis_space : int, optional  
        size of DIIS space for inner Fock loop, default 8
    fock_maxiter : int, optional
        maximum number of inner Fock loop iterations, default 50
    fock_maxruns : int, optional
        maximum number of outer Fock loop iterations, default 20
    ss_factor : float, optional
        same spin factor for auxiliary build, default 1.0
    os_factor : float, optional
        opposite spin factor for auxiliary build, default 1.0
    use_merge : bool, optional
        if True, perform the exact degeneracy-based merge, default
        False
    bath_type : str, optional
        GF truncation kernel method {'power', 'legendre'}, default 
        'power'
    bath_beta : int, optional
        inverse temperature used in GF truncation kernel, default 100

    Attributes
    ----------
    hf : UHF
        Hartree-Fock object
    nmom : tuple of int
        see parameters
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
        spins
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
    setup(uhf)
        constructs the object using the parameters provided to 
        `__init__` and performs the initial MP2 iteration
    get_fock(rdm1=None)
        returns the Fock matrix resulting from the current, or
        provided, density
    run()
        runs the method
    '''

    def __init__(self, uhf, **kwargs):
        self.hf = uhf
        self.options = _set_options(**kwargs)
        self._timer = util.Timer()

        self.setup()


    @util.record_time('setup')
    def setup(self):
        self.h1e = self.hf.h1e_mo
        self.eri = self.hf.eri_mo

        if self.options['dm0'] is None:
            self.rdm1 = self.hf.rdm1_mo
        else:
            self.rdm1 = np.asarray(self.options['dm0'], dtype=types.float64)

            if self.rdm1.ndim == 2:
                self.rdm1 = np.stack([self.rdm1, self.rdm1], axis=0)

        self.converged = False
        self.iteration = 0

        nact = self.hf.nao - self.options['frozen']
        self.se = (aux.Aux([], [[],]*nact, chempot=self.hf.chempot[0]),
                   aux.Aux([], [[],]*nact, chempot=self.hf.chempot[1]))
        self._se_prev = (None, None)

        self._timings = {}
        self._energies = {}

        log.title('Options', self.verbose)
        log.options(self.options, self.verbose)
        log.title('Input', self.verbose)
        log.molecule(self.hf.mol, self.verbose)
        log.write('Basis = %s\n' % self.hf.mol.basis, self.verbose)
        log.write('E(nuc) = %.12f\n' % self.hf.e_nuc, self.verbose)
        log.write('E(hf)  = %.12f\n' % self.hf.e_tot, self.verbose)
        log.write('<S^2> = %.6f\n' % self.hf.spin_square[0], self.verbose)
        log.write('nao = %d\n' % self.hf.nao, self.verbose)
        log.write('nmom = (%s, %s)\n' % self.nmom, self.verbose)

        self.run_mp2()


    @util.record_time('build')
    def build(self):
        self._se_prev = (self.se[0].copy(), self.se[1].copy())
        eri_act = _active(self, self.eri)

        if self.iteration:
            fock_act = _active(self, self.get_fock())
            sea, seb = aux.build_ump2_iter(self.se, fock_act, eri_act,
                                           **self.options['_build'])
        else:
            e_act = _active(self, self.hf.e)
            sea = aux.build_ump2(e_act, eri_act[0], **self.options['_build'])
            seb = aux.build_ump2(e_act[::-1], eri_act[1][::-1], 
                                 **self.options['_build'])

        if self.options['use_merge']:
            sea = sea.merge(etol=self.options['etol'], 
                            wtol=self.options['wtol'])
            seb = seb.merge(etol=self.options['etol'], 
                            wtol=self.options['wtol'])

        self.se = (sea, seb)

        log.write('naux (build,alpha) = %d\n' % (self.naux[0]), self.verbose)
        log.write('naux (build,beta)  = %d\n' % (self.naux[1]), self.verbose)


    @util.record_time('fock')
    def fock_loop(self):
        se, rdm1, converged = fock_loop_uhf(self.se, self.h1e, self.rdm1,
                                            self.eri, (self.nalph, self.nbeta),
                                            **self.options['_fock_loop'])

        if converged:
            log.write('Fock loop converged.\n', self.verbose)
            log.write('Chemical potential (alpha) = %.6f\n' % self.chempot[0],
                      self.verbose)
            log.write('Chemical potential (beta)  = %.6f\n' % self.chempot[1],
                      self.verbose)
        else:
            log.write('Fock loop did not converge.\n', self.verbose)
            log.write('Chemical potential (alpha) = %.6f\n' % self.chempot[0],
                      self.verbose)
            log.write('Chemical potential (beta)  = %.6f\n' % self.chempot[1],
                      self.verbose)

        h1e_act = _active(self, self.h1e)
        e_qmo_a = util.eigvalsh(se[0].as_hamiltonian(h1e_act[0]))
        e_qmo_b = util.eigvalsh(se[1].as_hamiltonian(h1e_act[1]))

        self.se = se
        self.rdm1 = rdm1

        log.write('HOQMO (alpha) = %.6f\n' % 
                  util.amax(e_qmo_a[e_qmo_a < self.chempot[0]]), self.verbose)
        log.write('HOQMO (beta)  = %.6f\n' % 
                  util.amax(e_qmo_b[e_qmo_b < self.chempot[1]]), self.verbose)
        log.write('LUQMO (alpha) = %.6f\n' % 
                  util.amin(e_qmo_a[e_qmo_a >= self.chempot[0]]), self.verbose)
        log.write('LUQMO (beta)  = %.6f\n' % 
                  util.amin(e_qmo_b[e_qmo_b >= self.chempot[1]]), self.verbose)
        log.array(self.rdm1[0], 'Density matrix (physical,alpha)', self.verbose)
        log.array(self.rdm1[1], 'Density matrix (physical,beta)', self.verbose)


    @util.record_time('merge')
    def merge(self):
        nmom_gf, nmom_se = self.nmom

        if nmom_gf is None and nmom_se is None:
            return

        fock_act = _active(self, self.get_fock())
        sea = self.se[0].compress(fock_act[0], self.nmom,
                                  method=self.options['bath_type'],
                                  beta=self.options['bath_beta'])
        seb = self.se[1].compress(fock_act[1], self.nmom,
                                  method=self.options['bath_type'],
                                  beta=self.options['bath_beta'])

        if self.options['use_merge']:
            sea = sea.merge(etol=self.options['etol'],
                            wtol=self.options['wtol'])
            seb = seb.merge(etol=self.options['etol'],
                            wtol=self.options['wtol'])

        self.se = (sea, seb)

        log.write('naux (merge,alpha) = %d\n' % self.naux[0], self.verbose)
        log.write('naux (merge,beta)  = %d\n' % self.naux[1], self.verbose)


    def damp(self):
        if self.options['damping'] == 0.0:
            return

        fcurr = np.sqrt(1.0 - self.options['damping'])
        fprev = np.sqrt(self.options['damping'])

        sea, seb = self.se
        sea_prev, seb_prev = self._se_prev

        sea._coup *= fcurr
        sea_prev._coup *= fprev
        sea = sea + sea_prev

        seb._coup *= fcurr
        seb_prev._coup *= fprev
        seb = seb + seb_prev

        self.se = (sea, seb)
        self._se_prev = (None, None)

        self.merge()


    def get_fock(self, rdm1=None):
        ''' Returns the Fock matrix resulting from the current, or
            provided, density.
        '''

        if rdm1 is None:
            rdm1 = self.rdm1

        fock = self.hf.get_fock(self.h1e, rdm1, self.eri)

        return fock


    @util.record_time('energy')
    @util.record_energy('mp2')
    def energy_mp2(self):
        e_act = _active(self, self.hf.e)
        emp2a = aux.energy.energy_mp2_aux(e_act[0], self.se[0])
        emp2b = aux.energy.energy_mp2_aux(e_act[1], self.se[1])

        emp2 = emp2a + emp2b
        emp2 /= 2

        log.write('E(mp2) = %.12f\n' % emp2, self.verbose)

        return emp2

    @util.record_time('energy')
    @util.record_energy('1b')
    def energy_1body(self):
        e1b = self.hf.energy_1body(self.h1e, self.rdm1, eri=self.eri)
        e1b += self.hf.mol.e_nuc

        log.write('E(1b)  = %.12f\n' % e1b, self.verbose)

        return e1b

    @util.record_time('energy')
    @util.record_energy('2b')
    def energy_2body(self):
        fock_act = _active(self, self.get_fock())
        gfa = aux.Aux(*self.se[0].eig(fock_act[0]), chempot=self.chempot[0])
        gfb = aux.Aux(*self.se[1].eig(fock_act[1]), chempot=self.chempot[0])

        e2ba = aux.energy.energy_2body_aux(gfa, self.se[0])
        e2bb = aux.energy.energy_2body_aux(gfb, self.se[1])

        e2b = e2ba + e2bb
        e2b /= 2

        log.write('E(2b)  = %.12f\n' % e2b, self.verbose)

        return e2b

    @util.record_time('energy')
    @util.record_energy('tot')
    def energy_tot(self):
        etot = self.e_1body + self.e_2body

        log.write('E(tot) = %12.f\n' % etot, self.verbose)

        return etot

    def energy(self):
        self.energy_1body()
        self.energy_2body()
        self.energy_tot()


    def run_mp2(self):
        log.iteration(0, self.verbose)

        self.build()
        self.merge()
        self.energy_mp2()


    def run(self):
        for self.iteration in range(1, self.options['maxiter']+1):
            log.iteration(self.iteration, self.verbose)

            self.fock_loop()
            self.build()
            self.merge()
            self.damp()
            self.energy()

            if self.iteration > 1:
                e_dif = abs(self._energies['tot'][-2] - \
                            self._energies['tot'][-1])

                if e_dif < self.options['etol'] and self.converged:
                    break

                self.converged = e_dif < self.options['etol']

        if self.converged:
            log.write('\nAuxiliary GF2 converged after %d iterations.\n' %
                      self.iteration, self.verbose)
        else:
            log.write('\nAuxiliary GF2 failed to converge.\n', self.verbose)

        self._timings['total'] = self._timings.get('total', 0.0) \
                                 + self._timer.total()
        log.title('Timings', self.verbose)
        log.timings(self._timings, self.verbose)

        return self.converged


    @property
    def nalph(self):
        return self.hf.nalph

    @property
    def nbeta(self):
        return self.hf.nbeta

    @property
    def nelec(self):
        return self.hf.nelec

    @property
    def nphys(self):
        return self.hf.nao

    @property
    def naux(self):
        return (self.se[0].naux, self.se[1].naux)

    @property
    def chempot(self):
        return (self.se[0].chempot, self.se[1].chempot)

    @property
    def e_hf(self):
        return self.hf.e_tot

    @property
    def e_1body(self):
        if self.iteration:
            return self._energies['1b'][-1]
        else:
            return self.e_hf

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
    def e_corr(self):
        return self.e_tot - self.e_hf

    @property
    def verbose(self):
        return self.options['verbose']

    @verbose.setter
    def verbose(self, val):
        self.options['verbose'] = val

    @property
    def nmom(self):
        return self.options['nmom']

    @nmom.setter
    def nmom(self, val):
        self.options['nmom'] = val


