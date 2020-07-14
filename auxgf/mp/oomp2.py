''' Orbital-optimized MP2 class.
'''

import numpy as np
from scipy.linalg import expm

from auxgf import util, hf
from auxgf.util import types


class OOMP2:
    def __init__(self, hf, maxiter=50, etol=1e-6, diis=True, diis_space=6, damping=0.5):
        self.hf = hf
        self.maxiter = maxiter
        self.etol = etol
        self.diis = diis
        self.diis_space = diis_space
        self.damping = damping

        self.e_1body = 0.0
        self.e_2body = 0.0
        self.e_prev = 0.0

        self.setup()

    def setup(self):
        self.eri_ao = util.spin_block(self.hf.eri_ao, self.hf.eri_ao)
        self.h1e_ao = util.spin_block(self.hf.h1e_ao, self.hf.h1e_ao)

        if isinstance(self.hf, hf.RHF):
            self.e = util.spin_block(self.hf.e, self.hf.e)
            self.c = util.spin_block(self.hf.c, self.hf.c)
        elif isinstance(self.hf, hf.UHF):
            self.e = util.spin_block(*list(self.hf.e))
            self.c = util.spin_block(*list(self.hf.c))

        mask = np.argsort(self.e)
        self.e = self.e[mask]
        self.c = self.c[:,mask]

        self.eri_ao = self.eri_ao.transpose(0,2,1,3) - \
                      self.eri_ao.transpose(0,2,3,1)

        self.h1e_mo = util.ao2mo(self.h1e_ao, self.c, self.c)
        self.eri_mo = util.ao2mo(self.eri_ao, self.c, self.c, self.c, self.c)

    def run(self):
        t_amp = np.zeros((self.nvir, self.nvir, self.nocc, self.nocc), dtype=types.float64)

        o = slice(None, self.nocc)
        v = slice(self.nocc, None)

        opdm_corr = np.zeros((self.nso,)*2, dtype=types.float64)
        opdm_ref  = opdm_corr.copy()
        opdm_ref[o,o] = np.eye(self.nocc)
        tpdm_corr = np.zeros((self.nso,)*4, dtype=types.float64)

        x = opdm_corr.copy()

        eija = util.outer_sum([-self.e[v], -self.e[v], self.e[o], self.e[o]])
        eija = 1.0 / eija

        if self.diis:
            diis = util.DIIS(self.diis_space)

        for niter in range(1, self.maxiter+1):
            f = self.h1e_mo + util.einsum('piqi->pq', self.eri_mo[:,o,:,o])
            fp = f.copy()
            np.fill_diagonal(fp, 0.0)
            e = f.diagonal()

            t1 = self.eri_mo[v,v,o,o]
            t2 = util.einsum('ac,cbij->abij', fp[v,v], t_amp)
            t3 = util.einsum('ki,abkj->abij', fp[o,o], t_amp)

            t_amp = t1.copy()
            t_amp += t2 - t2.transpose(1,0,2,3)
            t_amp -= t3 - t3.transpose(0,1,3,2)
            t_amp *= eija

            if niter > 1:
                if self.diis:
                    t_amp = diis.update(t_amp)

                if self.damping > 0.0:
                    damping = self.damping
                    t_amp = (1.0 - damping) * t_amp + damping * self._t_prev

            if self.damping > 0.0:
                self._t_prev = t_amp.copy()

            opdm_corr[v,v] = util.einsum('ijac,bcij->ba', t_amp.T, t_amp) * 0.5
            opdm_corr[o,o] = util.einsum('jkab,abik->ji', t_amp.T, t_amp) * -0.5
            opdm = opdm_corr + opdm_ref

            tpdm_corr[v,v,o,o] = t_amp
            tpdm_corr[o,o,v,v] = t_amp.T
            tpdm2 = util.einsum('rp,sq->rspq', opdm_corr, opdm_ref)
            tpdm3 = util.einsum('rp,sq->rspq', opdm_ref, opdm_ref)
            tpdm = tpdm_corr.copy()
            tpdm += tpdm2 - tpdm2.transpose(1,0,2,3)
            tpdm -= tpdm2.transpose(0,1,3,2) - tpdm2.transpose(1,0,3,2)
            tpdm += tpdm3 - tpdm3.transpose(1,0,2,3)

            fnr  = util.einsum('pr,rq->pq', self.h1e_mo, opdm)
            fnr += util.einsum('prst,stqr->pq', self.eri_mo, tpdm) * 0.5
            x[v,o] = ((fnr - fnr.T)[v,o]) / util.outer_sum([-self.e[v], self.e[o]])

            u = expm(x - x.T)
            c = np.dot(self.c, u)
            self.c = c

            self.h1e_mo = util.ao2mo(self.h1e_ao, c, c)
            self.eri_mo = util.ao2mo(self.eri_ao, c, c, c, c)

            e_prev = self.e_tot
            self.e_1body = util.einsum('pq,qp->', self.h1e_mo, opdm)
            self.e_1body += self.hf.e_nuc
            self.e_2body = 0.25 * util.einsum('pqrs,rspq->', self.eri_mo, tpdm)

            if abs(self.e_tot - e_prev) < self.etol:
                break

        return self

    @property
    def e_tot(self):
        return self.e_1body + self.e_2body

    @property
    def e_corr(self):
        return self.e_tot - self.hf.e_tot

    
    @property
    def nao(self):
        return self.hf.nao

    @property
    def nso(self):
        return self.nao * 2

    @property
    def nocc(self):
        return self.hf.nalph + self.hf.nbeta

    @property
    def nvir(self):
        return self.nso - self.nocc
