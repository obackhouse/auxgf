import unittest
import numpy as np
import scipy.special
import scipy.integrate

from auxgf import mol, hf, aux


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')
        self.rhf = hf.RHF(self.m).run()
        self.se = aux.build_rmp2(self.rhf.e, self.rhf.eri_mo, chempot=self.rhf.chempot)
        self.fock = self.rhf.fock_mo
        self.se = self.se.se_compress(self.fock, nmom=10)
        self.w, self.v = self.se.eig(self.fock)
        self.gf = self.se.new(self.w, self.v[:self.rhf.nao])

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.se, self.fock, self.w, self.v, self.gf

    def test_kernel_power(self):
        nmom = 5
        en1 = self.rhf.e[None,:] ** np.arange(nmom+1)[:,None]
        en2 = aux.gftrunc.kernel(self.rhf.e, nmom)
        for n in range(nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(en1 - en2)), 0, 12)

    def test_kernel_legendre(self):
        nmom = 8
        beta = 2**3

        def fn(t, i, n):
            p = scipy.special.legendre(n)
            x = 2 / beta * t + 1
            return p(x) * np.exp(-self.rhf.e[i] * (t + (self.rhf.e[i]>0) * beta))

        en1 = np.ones((nmom+1, self.rhf.nao))
        for n in range(1, nmom+1):
            for i in range(self.rhf.nao):
                en1[n,i] = scipy.integrate.quad(fn, -beta, 0, args=(i,n))[0]

        en2 = aux.gftrunc.kernel(self.rhf.e, nmom, method='legendre', beta=beta)

        for n in range(nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(en1 - en2)), 0, 6)

    def test_gftrunc_0(self):
        nmom = 0
        se_a = aux.gftrunc.run(self.se, self.fock, nmom)
        w_a, v_a = se_a.eig(self.fock)
        gf_a = se_a.new(w_a, v_a[:self.rhf.nao])

        m = self.gf.moment(range(2*nmom+1))
        m_a = gf_a.moment(range(2*nmom+1))
        for n in range(2*nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 8-n)

        m = self.gf.as_occupied().moment(range(2*nmom+1))
        m_a = gf_a.as_occupied().moment(range(2*nmom+1))
        for n in range(2*nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 8-n)

        m = self.gf.as_virtual().moment(range(2*nmom+1))
        m_a = gf_a.as_virtual().moment(range(2*nmom+1))
        for n in range(2*nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 8-n)

    def test_gftrunc_1(self):
        nmom = 1
        se_a = aux.gftrunc.run(self.se, self.fock, nmom)
        w_a, v_a = se_a.eig(self.fock)
        gf_a = se_a.new(w_a, v_a[:self.rhf.nao])

        m = self.gf.moment(range(2*nmom+1))
        m_a = gf_a.moment(range(2*nmom+1))
        for n in range(2*nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 8-n)

        m = self.gf.as_occupied().moment(range(2*nmom+1))
        m_a = gf_a.as_occupied().moment(range(2*nmom+1))
        for n in range(2*nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 8-n)

        m = self.gf.as_virtual().moment(range(2*nmom+1))
        m_a = gf_a.as_virtual().moment(range(2*nmom+1))
        for n in range(2*nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 8-n)

    def test_gftrunc_5(self):
        nmom = 5
        se_a = aux.gftrunc.run(self.se, self.fock, nmom)
        w_a, v_a = se_a.eig(self.fock)
        gf_a = se_a.new(w_a, v_a[:self.rhf.nao])

        m = self.gf.moment(range(2*nmom+1))
        m_a = gf_a.moment(range(2*nmom+1))
        for n in range(2*nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 8-n)

        m = self.gf.as_occupied().moment(range(2*nmom+1))
        m_a = gf_a.as_occupied().moment(range(2*nmom+1))
        for n in range(2*nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 8-n)

        m = self.gf.as_virtual().moment(range(2*nmom+1))
        m_a = gf_a.as_virtual().moment(range(2*nmom+1))
        for n in range(2*nmom+1):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 8-n)


if __name__ == '__main__':
    unittest.main()
