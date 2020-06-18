''' Class to perform second-order algebraic diagrammatric construction
    theory for restricted references.

    Derivation and algorithm due to Banerjee & Sokolov.
'''

import numpy as np
import scipy.sparse.linalg as sl

from auxgf import util, aux
from auxgf.util import types, log, mpi
from auxgf.adc import radc2

#TODO: subclass? Fock and GF defined?


_set_options = radc2._set_options


class UADC2(util.AuxMethod):
    ''' Class for second-order algebraic diagrammatic construction.

    Parameters
    ----------
    hf : UHF
        Hartree-Fock object
    nmom : tuple of int, optional
        number of moments to which the truncation is constistent to,
        ordered by (Green's function, self-energy), default is
        (None, None) which means no truncation (full MP2).
    verbose : bool, optional
        if True, print output log, default True
    method : str, optional
        which excitation to calculate, 'ip' or 'ea', default 'ip'
    nroots : int, optional
        number of excitations to calculate, default 1
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


    def _get_slices(self):
        oa = slice(None, self.hf.nalph)
        va = slice(self.hf.nalph, None)
        ob = slice(None, self.hf.nbeta)
        vb = slice(self.hf.nbeta, None)

        if self.method == 'ip':
            return oa,ob,va,vb
        else:
            return va,vb,oa,ob



    def _get_sizes(self):
        nocca, noccb = self.hf.nalph, self.hf.nbeta
        nvira, nvirb = self.nphys - nocca, self.nphys - noccb

        if self.method == 'ip':
            return nocca, noccb, nvira, nvirb
        else:
            return nvira, nvirb, nocca, noccb


    @util.record_time('1p/1h')
    def get_1p_or_1h(self):
        occa, occb, vira, virb = self._get_slices()

        self.e_mp2 = 0
        hs = []

        for a,b in [(0,1), (1,0)]:
            if a == 0:
                occa, occb, vira, virb = self._get_slices()
            else:
                occb, occa, virb, vira = self._get_slices()

            eri_aa_ovov = self.eri[a,a][occa,vira,occa,vira]
            eri_ab_ovov = self.eri[a,b][occa,vira,occb,virb]

            eo_a = self.hf.e[a][occa]
            eo_b = self.hf.e[b][occb]
            ev_a = self.hf.e[a][vira]
            ev_b = self.hf.e[b][virb]

            t2_aa = eri_aa_ovov.copy()
            t2_aa /= util.dirsum('i,a,j,b->iajb', eo_a, -ev_a, eo_a, -ev_a)
            t2a_aa = t2_aa - t2_aa.swapaxes(0,2).copy()

            t2_ab = eri_ab_ovov.copy()
            t2_ab /= util.dirsum('i,a,j,b->iajb', eo_a, -ev_a, eo_b, -ev_b)

            self.e_mp2 += util.einsum('iajb,iajb->', t2a_aa, eri_aa_ovov) * 0.5
            self.e_mp2 += util.einsum('iajb,iajb->', t2_ab, eri_ab_ovov) * 0.5

            h = np.diag(eo_a)

            h += util.einsum('a,iakb,jakb->ij', ev_a, t2a_aa, t2a_aa)
            h += util.einsum('a,iakb,jakb->ij', ev_a, t2_ab, t2_ab)
            h += util.einsum('a,ibka,jbka->ij', ev_b, t2_ab, t2_ab)
            h -= util.einsum('k,iakb,jakb->ij', eo_a, t2a_aa, t2a_aa) * 0.5
            h -= util.einsum('k,iakb,jakb->ij', eo_b, t2_ab, t2_ab) * 0.5
            h -= util.einsum('k,iakb,jakb->ij', eo_b, t2_ab, t2_ab) * 0.5
            h -= util.einsum('i,iakb,jakb->ij', eo_a, t2a_aa, t2a_aa) * 0.25
            h -= util.einsum('i,iakb,jakb->ij', eo_a, t2_ab, t2_ab) * 0.25
            h -= util.einsum('i,iakb,jakb->ij', eo_a, t2_ab, t2_ab) * 0.25
            h -= util.einsum('j,iakb,jakb->ij', eo_a, t2a_aa, t2a_aa) * 0.25
            h -= util.einsum('j,iakb,jakb->ij', eo_a, t2_ab, t2_ab) * 0.25
            h -= util.einsum('j,iakb,jakb->ij', eo_a, t2_ab, t2_ab) * 0.25
            h += util.einsum('iakb,jakb->ij', t2a_aa, eri_aa_ovov) * 0.5
            h -= util.einsum('iakb,jbka->ij', t2a_aa, eri_aa_ovov) * 0.5
            h += util.einsum('iakb,jakb->ij', t2_ab, eri_ab_ovov)
            h += util.einsum('jakb,iakb->ij', t2a_aa, eri_aa_ovov) * 0.5
            h -= util.einsum('jakb,kaib->ij', t2a_aa, eri_aa_ovov) * 0.5
            h += util.einsum('jakb,iakb->ij', t2_ab, eri_ab_ovov)

            hs.append(h)

        self.h_1p_or_1h = tuple(hs)


    @util.record_time('2p1h/2h1p')
    def build(self):
        se = []

        for a,b in [(0,1), (1,0)]:
            if a == 0:
                occa, occb, vira, virb = self._get_slices()
            else:
                occb, occa, virb, vira = self._get_slices()

            eri_aa_ooov = self.eri[a,a][occa,occa,occa,vira]
            eri_ab_ooov = self.eri[a,b][occa,occa,occb,virb]

            eo_a = self.hf.e[a][occa]
            eo_b = self.hf.e[b][occb]
            ev_a = self.hf.e[a][vira]
            ev_b = self.hf.e[b][virb]

            eo = (eo_a, eo_b)
            ev = (ev_a, ev_b)
            xija = (eri_aa_ooov, eri_ab_ooov)
            
            e, v = aux.build_ump2_part(eo, ev, xija, **self.options['_build'])

            se.append(aux.Aux(e, v))

        self.se = tuple(se)

        log.write('naux (build,alpha) = %d\n' % (self.naux[0]), self.verbose)
        log.write('naux (build,beta)  = %d\n' % (self.naux[1]), self.verbose)


    @util.record_time('merge')
    def merge(self):
        nmom_gf, nmom_se = self.nmom
        se = list(self.se)

        for a,b in [(0,1), (1,0)]:
            if a == 0:
                occa, occb, vira, virb = self._get_slices()
            else:
                occb, occa, virb, vira = self._get_slices()

            fock = self.get_fock()[a][occa,occa]
            se[a] = self.se[a].compress(fock, self.nmom, 
                                        **self.options['_merge'])

        self.se = tuple(se)

        log.write('naux (merge,alpha) = %d\n' % self.naux[0], self.verbose)
        log.write('naux (merge,beta)  = %d\n' % self.naux[1], self.verbose)


    @util.record_time('diagonalise')
    def diagonalise(self):
        self.e_excite = []
        self.v_excite = []

        for a,b in [(0,1), (1,0)]:
            if a == 0:
                occa, occb, vira, virb = self._get_slices()
            else:
                occb, occa, virb, vira = self._get_slices()

            w, v = self.se[a].eig(self.h_1p_or_1h[a], nroots=1)

            self.e_excite.append(-w if self.method == 'ip' else w)
            self.v_excite.append(v)


    def run(self):
        self.get_1p_or_1h()
        self.build()
        self.merge()
        self.diagonalise()

        log.write('E(mp2) = %.12f\n' % self.e_mp2, self.verbose)
        e_excite = self.ip[0] if self.method == 'ip' else self.ea[0]
        log.write('E(%s)  = %.12f\n' % (self.method, e_excite), 
                                       self.verbose)

        self._timings['total'] = self._timings.get('total', 0.0) \
                                 + self._timer.total()
        log.title('Timings', self.verbose)
        log.timings(self._timings, self.verbose)

        return self


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
    def ip(self):
        if self.method != 'ip':
            raise ValueError
        if self.e_excite[0][0] > self.e_excite[1][0]:
            return self.e_excite[0][0], self.v_excite[0][0]
        else:
            return self.e_excite[1][0], self.v_excite[1][0]

    @property
    def ea(self):
        if self.method != 'ea':
            raise ValueError
        if self.e_excite[0][0] < self.e_excite[1][0]:
            return self.e_excite[0][0], self.v_excite[0][0]
        else:
            return self.e_excite[1][0], self.v_excite[1][0]


    @property
    def method(self):
        return self.options['method']

    @method.setter
    def method(self, val):
        self.options['method'] = val

    @property
    def nmom(self):
        return self.options['nmom']

    @nmom.setter
    def nmom(self, val):
        self.options['nmom'] = val
