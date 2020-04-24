import numpy as np

from auxgf import util, aux
from auxgf.util import types, log, mpi
from auxgf.agf2.fock import fock_loop_rhf


#TODO definitely rethink this algorithm

def _set_options(**kwargs):
    options = { 'scheme' : 'G0W0',
                'nmom' : (3,4),
                'dm0' : None,
                'verbose' : True,
                'maxiter' : 50,
                'etol' : 1e-6,
                'damping' : 0.5,
                'dtol' : 1e-8,
                'diis_space' : 8,
                'fock_maxiter' : 50,
                'fock_maxruns' : 20,
                'qp_maxiter' : 50,
                'qp_etol' : 1e-6,
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
        'verbose' : options['verbose'],
    }

    if options['scheme'] not in ['G0W0', 'GW0', 'GW', 'evGW0', 'evGW']:
        raise ValueError('%s scheme not recognised.' % options['scheme'])

    options['update_eig'] = 'ev' in options['scheme']
    options['update_w'] = 'W0' not in options['scheme']

    if options['scheme'] == 'G0W0':
        options['maxiter'] = 1

    options['verbose'] = options['verbose'] and mpi.rank

    return options


def RAG0W0(*args, **kwargs): return RAGWA(*args, scheme='G0W0', **kwargs)
def RAGW0(*args, **kwargs): return RAGWA(*args, scheme='GW0', **kwargs)
def RAGW(*args, **kwargs): return RAGWA(*args, scheme='GW', **kwargs)
def RevGW0(*args, **kwargs): return RAGWA(*args, scheme='evGW0', **kwargs)
def RevGW(*args, **kwargs): return RAGWA(*args, scheme='evGW', **kwargs)


class RAGWA:
    ''' Restricted auxiliary GW approximation.

    Parameters
    ----------
    rhf : RHF
        Hartree-Fock object
    nmom : tuple of int, optional
        number of moments to which to truncation is consistent to,
        ordered by (Green's function, self-energy), default (3,4),
        see auxgf.aux.aux.Aux.compress
    dm0 : (n,n) ndarray, optional
        initial density matrix, if None, use rhf.rdm1_mo, default
        None
    verbose : bool, optional
        if True, print output log, default True
    maxiter : int, optional
        maximum number of RAGF2 iterations, default 50
    etol : float, optional
        maximum difference in subsequent energies at convergence,
        default 1e-6
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
    qp_maxiter : int, optional
        maximum number of quasiparticle iterations, default 50
    qp_etol : float, optional
        maximum difference in subsequent quasiparticle energies at
        convergence, default 1e-6

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
    '''

    def __init__(self, rhf, **kwargs):
        self.hf = rhf
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
            self.rdm1 = np.array(self.options['dm0'], dtype=types.float64)

        self.converged = False
        self.iteration = 0

        chempot = self.hf.chempot
        self.gf = aux.Aux(self.hf.e, np.eye(self.hf.nao), chempot=chempot)
        self.se = aux.Aux([], [[],]*self.hf.nao, chempot=chempot)
        self._se_prev = None
        self.rpa = None
        self.e_qp = self.hf.e.copy()

        self._timings = {}
        self._energies = {}

        log.title('Options', self.verbose)
        log.options(self.options, self.verbose)
        log.title('Input', self.verbose)
        log.molecule(self.hf.mol, self.verbose)
        log.write('Basis = %s\n' % self.hf.mol.basis, self.verbose)
        log.write('E(nuc) = %.12f\n' % self.hf.e_nuc, self.verbose)
        log.write('E(hf)  = %.12f\n' % self.hf.e_tot, self.verbose)
        log.write('nao = %d\n' % self.hf.nao, self.verbose)
        log.write('nmom = (%s, %s)\n' % self.nmom, self.verbose)


    @util.record_time('gf')
    def build_gf(self):
        e, c = self.se.eig(self.get_fock())

        self.gf = aux.Aux(e, c, chempot=self.chempot)

        c_occ = c[:self.nphys, e < self.chempot]
        self.rdm1 = np.dot(c_occ, c_occ.T) * 2

        log.write('HOQMO = %.6f\n' % 
                  util.amax(self.gf.e[self.gf.e < self.chempot]), self.verbose)
        log.write('LUQMO = %.6f\n' %
                  util.amin(self.gf.e[self.gf.e >= self.chempot]), self.verbose)
        log.array(self.rdm1, 'Density matrix (physical)', self.verbose)

        return self.gf


    @util.record_time('rpa')
    def solve_casida(self):
        #TODO: this step is n^6 and inefficient in memory, rethink

        e_ia = util.outer_sum([-self.gf.e_occ, self.gf.e_vir])

        co = self.gf.v[:self.nphys, self.gf.e < self.chempot]
        cv = self.gf.v[:self.nphys, self.gf.e >= self.chempot]
        iajb = util.ao2mo(self.eri, co, cv, co, cv).reshape((e_ia.size,)*2)

        apb = np.diag(e_ia.flatten()) + 4.0 * iajb
        amb = np.diag(np.sqrt(e_ia.flatten()))

        h_rpa = util.dots((amb, apb, amb))
        e_rpa, v_rpa = util.eigh(h_rpa)
        e_rpa = np.sqrt(e_rpa)

        xpy = util.einsum('ij,jk,k->ik', amb, v_rpa, 1.0 / np.sqrt(e_rpa))
        xpy *= np.sqrt(2.0)

        self.rpa = (e_rpa, v_rpa, xpy)

        return self.rpa


    @util.record_time('se')
    def build_se(self):
        if self.rpa is None:
            self.solve_casida()

        e_rpa, v_rpa, xpy = self.rpa
        naux_gf = self.gf.naux
        chempot = self.chempot

        c = self.gf.v[:self.nphys]
        co = c[:, self.gf.e < chempot]
        cv = c[:, self.gf.e >= chempot]
        xyia = util.mo2qo(self.eri, c, co, cv).reshape(self.nphys, naux_gf, -1)

        omega = util.einsum('xyk,ks->xys', xyia, xpy)
        e_gf = self.gf.e

        e_rpa_s = np.outer(np.sign(e_gf - chempot), e_rpa)
        e = util.dirsum('i,ij->ij', e_gf, e_rpa_s).flatten()
        v = omega.reshape((self.nphys, -1))

        self.se = aux.Aux(e, v, chempot=self.chempot)

        log.write('naux (se,build) = %d\n' % self.se.naux, self.verbose)

        return self.se


    @util.record_time('fock')
    def fock_loop(self):
        se, rdm1, converged = fock_loop_rhf(self.se, self.h1e, self.rdm1,
                                            self.eri, self.nelec,
                                            **self.options['_fock_loop'])

        if converged:
            log.write('Fock loop converged.\n', self.verbose)
            log.write('Chemical potential = %.6f\n' % self.chempot,
                      self.verbose)
        else:
            log.write('Fock loop did not converge.\n', self.verbose)
            log.write('Chemical potential = %.6f\n' % self.chempot,
                      self.verbose)

        self.se = se
        self.gf.chempot = se.chempot
        self.build_gf()

        return converged


    @util.record_time('se') # kind of
    def iterate_qp(self):
        # Should e_qp become e_gf or is that only for evGW?

        e_qp = self.e_qp
        e_qp_0 = e_qp.copy()

        e = self.gf.e
        v = self.gf.v

        for niter in range(1, self.options['qp_maxiter']+1):
            denom = 1.0 / util.outer_sum(e_qp, -e)
            se = v * v * denom
            z = se * denom

            se = np.sum(se, axis=1)
            z = np.sum(z, axis=1)
            z = 1.0 / (1.0 + z)

            e_qp_prev = e_qp.copy()
            e_qp = e_qp_0 + z.real * se
            error = np.max(np.absolute(e_qp.real - e_qp_prev.real))

            if error < self.options['qp_etol']:
                break

        self.e_qp = e_qp
        
        return e_qp


    @util.record_time('merge')
    def merge(self):
        nmom_gf, nmom_se = self.nmom

        if nmom_gf is None and nmom_se is None:
            return

        self.se = self.se.compress(self.get_fock(), self.nmom)

        log.write('naux (se,merge) = %d\n' % self.se.naux, self.verbose)


    def damp(self):
        if self.options['damping'] == 0.0:
            return

        fcurr = np.sqrt(1.0 - self.options['damping'])
        fprev = np.sqrt(self.options['damping'])

        self.se._coup *= fcurr
        self._se_prev._coup *= fprev

        self.se = self.se + self._se_prev
        self._se_prev = None

        self.merge()


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


    def run(self):
        for self.iteration in range(1, self.options['maxiter']+1):
            log.iteration(self.iteration, self.verbose)

            self.build_gf()

            if self.iteration == 1 or self.options['update_w']:
                self.solve_casida()

            self.build_se()
            self.merge()
            self.energy()

            if self.options['update_eig']:
                self.iterate_qp()

            if self.iteration > 1:
                e_dif = abs(self._energies['tot'][-2] - \
                            self._energies['tot'][-1])

                if e_dif < self.options['etol'] and self.converged:
                    break

                self.converged = e_dif < self.options['etol']

        if self.converged:
            log.write('\nAuxiliary GW converged after %d iterations.\n' %
                      self.iteration, self.verbose)
        else:
            log.write('\nAuxiliary GW failed to converge.\n', self.verbose)

        self._timings['total'] = self._timings.get('total', 0.0) \
                                 + self._timer.total()
        log.title('Timings', self.verbose)
        log.timings(self._timings, self.verbose)

        return self


    def get_fock(self, rdm1=None):
        ''' Returns the Fock matrix resulting from the current, or
            provided, density.
        '''

        if rdm1 is None:
            rdm1 = self.rdm1

        return self.hf.get_fock(rdm1, basis='mo')

    
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
        return (self.gf.naux, self.se.naux)

    @property
    def chempot(self):
        return self.gf.chempot
    
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
            return 0.0

    @property
    def e_tot(self):
        if self.iteration:
            return self._energies['tot'][-1]
        else:
            return self.e_hf

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


