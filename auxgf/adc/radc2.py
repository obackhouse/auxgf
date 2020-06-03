''' Class to perform second-order algebraic diagrammatric construction
    theory for restricted references.

    Derivation and algorithm due to Banerjee & Sokolov.
'''

import numpy as np
import scipy.sparse.linalg as sl

from auxgf import util, aux
from auxgf.util import types, log, mpi


def _set_options(**kwargs):
    options = { 'method' : 'ip', 
                'nroots' : 1,
                'verbose' : True,
                'wtol' : 1e-12,
                'ss_factor' : 1.0,
                'os_factor' : 1.0,
    }

    for key,val in kwargs.items():
        if key not in options.keys():
            raise ValueError('%s argument invalid.' % key)

    options.update(kwargs)

    options['_build'] = {
        'wtol' : options['wtol'],
        'ss_factor' : options['ss_factor'],
        'os_factor' : options['os_factor'],
    }

    options['verbose'] = options['verbose'] and not mpi.rank

    return options


class RADC2:
    ''' Class for second-order algebraic diagrammatic construction.

    Parameters
    ----------
    hf : RHF or UHF
        Hartree-Fock object
    method : str, optional
        which excitation to calculate, 'ip' or 'ea', default 'ip'
    nroots : int, optional
        number of excitations to calculate, default 1

    Attributes
    ----------
    e_tot : float
        total energy
    e_corr : float
        correlation energy
    e_hf : float
        Hartree-Fock energy
    ip : float, ndarray of floats
        ionization potential
    ea : float, ndarray of floats
        electron affinity

    Methods
    -------
    run(*args, **kwargs)
        runs the calculation (performed automatically), see class
        parameters for arguments
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
        self.rdm1 = self.hf.rdm1_mo

        self._timings = {}

        log.title('Options', self.verbose)
        log.options(self.options, self.verbose)
        log.title('Input', self.verbose)
        log.molecule(self.hf.mol, self.verbose)
        log.write('Basis = %s\n' % self.hf.mol.basis, self.verbose)
        log.write('E(nuc) = %.12f\n' % self.hf.e_nuc, self.verbose)
        log.write('E(hf)  = %.12f\n' % self.hf.e_tot, self.verbose)
        log.write('nao = %d\n' % self.hf.nao, self.verbose)


    def _get_slices(self):
        o = slice(None, self.hf.nocc)
        v = slice(self.hf.nocc, None)

        if self.method == 'ip':
            return o,v
        else:
            return v,o


    def _get_sizes(self):
        if self.method == 'ip':
            return self.hf.nocc, self.hf.nvir
        else:
            return self.hf.nvir, self.hf.nocc


    @util.record_time('1p/1h')
    def get_1p_or_1h(self):
        occ, vir = self._get_slices()

        eri_ovov = self.eri[occ,vir,occ,vir]
        eo = self.hf.e[occ]
        ev = self.hf.e[vir]

        t2 = eri_ovov.copy()
        t2 /= util.dirsum('i,a,j,b->iajb', eo, -ev, eo, -ev)
        t2a = t2 - t2.swapaxes(0,2).copy()

        self.e_mp2  = util.einsum('iajb,iajb->', t2, eri_ovov) * 2
        self.e_mp2 -= util.einsum('iajb,ibja->', t2, eri_ovov)

        h = np.diag(eo)

        h += util.einsum('a,iakb,jakb->ij', ev, t2a, t2a)
        h += util.einsum('a,iakb,jakb->ij', ev, t2, t2)
        h += util.einsum('a,ibka,jbka->ij', ev, t2, t2)
        h -= util.einsum('k,iakb,jakb->ij', eo, t2a, t2a) * 0.5
        h -= util.einsum('k,iakb,jakb->ij', eo, t2, t2) * 0.5
        h -= util.einsum('k,iakb,jakb->ij', eo, t2, t2) * 0.5
        h -= util.einsum('i,iakb,jakb->ij', eo, t2a, t2a) * 0.25
        h -= util.einsum('i,iakb,jakb->ij', eo, t2, t2) * 0.25
        h -= util.einsum('i,iakb,jakb->ij', eo, t2, t2) * 0.25
        h -= util.einsum('j,iakb,jakb->ij', eo, t2a, t2a) * 0.25
        h -= util.einsum('j,iakb,jakb->ij', eo, t2, t2) * 0.25
        h -= util.einsum('j,iakb,jakb->ij', eo, t2, t2) * 0.25
        h += util.einsum('iakb,jakb->ij', t2a, eri_ovov) * 0.5
        h -= util.einsum('iakb,jbka->ij', t2a, eri_ovov) * 0.5
        h += util.einsum('iakb,jakb->ij', t2, eri_ovov)
        h += util.einsum('jakb,iakb->ij', t2a, eri_ovov) * 0.5
        h -= util.einsum('jakb,kaib->ij', t2a, eri_ovov) * 0.5
        h += util.einsum('jakb,iakb->ij', t2, eri_ovov)

        self.h_1p_or_1h = h


    @util.record_time('2p1h/2h1p')
    def build(self):
        occ, vir = self._get_slices()

        eri_ooov = self.eri[occ,occ,occ,vir]
        eo = self.hf.e[occ]
        ev = self.hf.e[vir]

        e, v = aux.build_rmp2_part(eo, ev, eri_ooov, 
                                   **self.options['_build'])

        self.se = aux.Aux(e, v, chempot=self.hf.chempot)


    @util.record_time('diagonalise')
    def diagonalise(self):
        nocc, nvir = self._get_sizes()

        matvec = lambda x : self.se.dot(self.h_1p_or_1h, x)
        linop = sl.LinearOperator(shape=(self.nphys+self.naux,)*2, 
                                  dtype=types.float64, matvec=matvec)

        which = 'LA' if self.method == 'ip' else 'SA'
        w, v = sl.eigsh(linop, k=self.options['nroots'], which=which)

        self.e_excite = -w if self.method == 'ip' else w
        self.v_excite = v


    def run(self):
        self.get_1p_or_1h()
        self.build()
        self.diagonalise()

        log.write('E(mp2) = %.12f\n' % self.e_mp2, self.verbose)
        log.write('E(%s)  = %.12f\n' % (self.method, self.e_excite[0]), 
                                       self.verbose)

        self._timings['total'] = self._timings.get('total', 0.0) \
                                 + self._timer.total()
        log.title('Timings', self.verbose)
        log.timings(self._timings, self.verbose)

        return self


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
        return self.se.nphys

    @property
    def naux(self):
        return self.se.naux

    @property
    def chempot(self):
        return self.se.chempot

    @property
    def e_hf(self):
        return self.hf.e_tot

    @property
    def e_1body(self):
        return self.e_hf

    @property
    def e_2body(self):
        return self.e_mp2

    @property
    def e_tot(self):
        return self.e_hf + self.e_mp2

    @property
    def e_corr(self):
        return self.e_tot - self.e_hf

    @property
    def ip(self):
        if self.method != 'ip':
            raise ValueError
        return self.e_excite, self.v_excite

    @property
    def ea(self):
        if self.method != 'ea':
            raise ValueError
        return self.e_excite, self.v_excite

    @property
    def verbose(self):
        return self.options['verbose']

    @verbose.setter
    def verbose(self, val):
        self.options['verbose'] = val

    @property
    def method(self):
        return self.options['method']

    @method.setter
    def method(self, val):
        self.options['method'] = val
