''' Class to perform auxiliary GF2 calculations for restricted
    references.
'''

import numpy as np
import functools

from auxgf import util, aux
from auxgf.util import types, log, mpi
from auxgf.agf2.fock import fock_loop_rhf


def _set_options(options, **kwargs):
    options.update({ 'nmom' : (3,4),
                     'maxiter' : 50,
                     'etol' : 1e-6,
                     'wtol' : 1e-12,
                     'damping' : 0.0,
                     'delay_damping' : 0,
                     'dtol' : 1e-8,
                     'diis_space' : 8,
                     'fock_maxiter' : 50,
                     'fock_maxruns' : 20,
                     'ss_factor' : 1.0,
                     'os_factor' : 1.0,
                     'use_merge' : False,
                     'bath_type' : 'power',
                     'bath_beta' : 100,
                     'qr' : 'cholesky',
    })

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


def _active(ragf2, arr, ndim):
    ''' Returns the active space of an n-dimensional array.
    '''

    frozen = ragf2.options['frozen']

    if not frozen:
        return arr

    act_ = slice(frozen[0], arr.shape[0]-frozen[1])

    act = (Ellipsis,) + (act_,)*ndim

    return arr[act]


class RAGF2(util.AuxMethod):
    ''' Restricted auxiliary GF2 method.

    Parameters
    ----------
    rhf : RHF
        Hartree-Fock object
    nmom : tuple of int, optional
        number of moments to which the truncation is consistent to,
        ordered by (Green's function, self-energy), default (3,4), 
        see auxgf.aux.aux.Aux.compress
    dm0 : (n,n) ndarray, optional
        initial density matrix, if None, use rhf.rdm1_mo, default
        None
    frozen : int or tuple of (int, int), optional
        number of frozen core orbitals, if tuple then second element 
        defines number of frozen virtual orbitals, default 0
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
    qr : str, optional
        type of QR solver to use for SE truncation {'cholesky', 
        'numpy', 'scipy', 'unsafe'}, default 'cholesky'

    Attributes
    ----------
    hf : RHF
        Hartree-Fock object
    nmom : tuple of int
        see parameters
    verbose : bool
        see parameters
    options : dict
        dictionary of options
    rdm1 : (n,n) ndarray
        one-particle reduced density matrix, projected into the
        physical basis
    converged : bool
        whether the method has converged
    iteration : int
        the current/final iteration reached
    se : Aux
        auxiliary representation of the self-energy
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

    def __init__(self, rhf, **kwargs):
        super().__init__(rhf, **kwargs)

        self.options = _set_options(self.options, **kwargs)

        self.setup()


    @util.record_time('setup')
    def setup(self):
        super().setup()

        log.title('Options', self.verbose)
        log.options(self.options, self.verbose)
        log.title('Input', self.verbose)
        log.molecule(self.hf.mol, self.verbose)
        log.write('Basis = %s\n' % self.hf.mol.basis, self.verbose)
        log.write('E(nuc) = %.12f\n' % self.hf.e_nuc, self.verbose)
        log.write('E(hf)  = %.12f\n' % self.hf.e_tot, self.verbose)
        log.write('nao = %d\n' % self.hf.nao, self.verbose)
        log.write('nfrozen = (%d, %d)\n' % self.options['frozen'], self.verbose)
        log.write('nmom = (%s, %s)\n' % self.nmom, self.verbose)

        self.run_mp2()


    @util.record_time('build')
    def build(self):
        build_opts = self.options['_build']
        etol = self.options['etol']
        wtol = self.options['wtol']
        use_merge = self.options['use_merge']

        self._se_prev = self.se.copy()
        eri_act = _active(self, self.eri, 2 if self.hf.with_df else 4)
        fock_act = _active(self, self.get_fock(), 2)

        if self.hf.with_df:
            self.se = aux.build_dfmp2_iter(self.se, fock_act, eri_act, **build_opts)
        else:
            self.se = aux.build_mp2_iter(self.se, fock_act, eri_act, **build_opts)
        
        if use_merge:
            self.se = self.se.merge(etol=etol, wtol=wtol)

        log.write('naux (build) = %d\n' % self.naux, self.verbose)


    @util.record_time('fock')
    def fock_loop(self):
        fock_opts = self.options['_fock_loop']
        fock_opts['fock_func'] = self.get_fock

        se, rdm1, converged = fock_loop_rhf(self.se, self.hf, self.rdm1, **fock_opts)

        if converged:
            log.write('Fock loop converged.\n', self.verbose)
            log.write('Chemical potential = %.6f\n' % self.chempot, self.verbose)
        else:
            log.write('Fock loop did not converge.\n', self.verbose)
            log.write('Chemical potential = %.6f\n' % self.chempot, self.verbose)

        fock_act = _active(self, self.get_fock(rdm1=rdm1), 2)

        e_qmo, v_qmo = util.eigh(se.as_hamiltonian(fock_act))
        self.gf = self.se.new(e_qmo, v_qmo[:self.nphys])

        self.se = se
        self.rdm1 = rdm1

        e_hoqmo = util.amax(e_qmo[e_qmo < self.chempot])
        e_luqmo = util.amin(e_qmo[e_qmo >= self.chempot])

        log.write('HOQMO = %.6f\n' % e_hoqmo, self.verbose)
        log.write('LUQMO = %.6f\n' % e_luqmo, self.verbose)
        log.array(self.rdm1, 'Density matrix (physical)', self.verbose)


    @util.record_time('merge')
    def merge(self):
        etol = self.options['etol']
        wtol = self.options['wtol']
        use_merge = self.options['use_merge']
        method=self.options['bath_type']
        beta=self.options['bath_beta']
        qr=self.options['qr']

        nmom_gf, nmom_se = self.nmom

        if nmom_gf is None and nmom_se is None:
            return

        fock = _active(self, self.get_fock(), 2)

        self.se = self.se.compress(fock, self.nmom, method=method, beta=beta, qr=qr)

        if self.options['use_merge']:
            self.se = self.se.merge(etol=etol, wtol=wtol)                                    

        log.write('naux (merge) = %d\n' % self.naux, self.verbose)


    def damp(self):
        damping = self.options['damping']
        delay_damping = self.options['delay_damping']

        if damping == 0.0:
            return

        if self.iteration <= delay_damping:
            return

        fcurr = np.sqrt(1.0 - damping)
        fprev = np.sqrt(damping)

        self.se._coup *= fcurr
        self._se_prev._coup *= fprev

        self.se = self.se + self._se_prev
        self._se_prev = None

        self.merge()


    @util.record_time('energy')
    @util.record_energy('mp2')
    def energy_mp2(self):
        emp2 = aux.energy.energy_mp2_aux(_active(self, self.hf.e, 1), self.se)

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
        fock_act = _active(self, self.get_fock(), 2)
        e_qmo, v_qmo = self.se.eig(fock_act)
        self.gf = self.se.new(e_qmo, v_qmo[:self.nphys])

        e2b = aux.energy.energy_2body_aux(self.gf, self.se)

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


    def run_mp2(self):
        log.iteration(0, self.verbose)

        self.build()
        self.merge()
        self.energy_mp2()


    def run(self):
        maxiter = self.options['maxiter']
        etol = self.options['etol']

        for self.iteration in range(1, maxiter+1):
            log.iteration(self.iteration, self.verbose)

            self.fock_loop()
            self.build()
            self.merge()
            self.damp()
            self.energy()

            if self.iteration > 1:
                e_dif = abs(self._energies['tot'][-2] - self._energies['tot'][-1])

                if e_dif < etol and self.converged:
                    break

                self.converged = e_dif < etol

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
        return self.options['nmom']

    @nmom.setter
    def nmom(self, val):
        self.options['nmom'] = val
