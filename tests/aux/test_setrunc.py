import unittest
import numpy as np

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

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.se, self.fock

    def test_qr(self):
        m = np.random.random((self.rhf.nao, self.rhf.nao)) - 0.5
        m = np.dot(m, m.T)
        a1, b1 = np.linalg.qr(m)
        a2, b2 = aux.setrunc._get_qr_function(method='cholesky')(m)
        a3, b3 = aux.setrunc._get_qr_function(method='numpy')(m)
        a4, b4 = aux.setrunc._get_qr_function(method='scipy')(m)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(a1, b1) - m)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(a2, b2) - m)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(a3, b3) - m)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(a4, b4) - m)), 0, 8)

    def test_build_block_tridiag(self):
        m1, m2, m3, b1, b2 = [np.random.random((self.rhf.nao, self.rhf.nao)) for x in range(5)]
        z = np.zeros((self.rhf.nao, self.rhf.nao))
        a1 = np.block([[m1, b1.T, z], [b1, m2, b2.T], [z, b2, m3]])
        a2 = aux.setrunc.build_block_tridiag([m1, m2, m3], [b1, b2])
        self.assertAlmostEqual(np.max(np.absolute(a1 - a2)), 0, 8)

    def test_block_lanczos_debug(self):
        aux.setrunc.band_lanczos(self.se.as_occupied(), self.fock, 2, debug=True)

    def test_band_lanczos_debug(self):
        aux.setrunc.band_lanczos(self.se.as_occupied(), self.fock, 2, debug=True)

    def test_setrunc_0(self):
        nmom = 0
        se_a = aux.setrunc.run(self.se, self.fock, nmom, method='block', qr='cholesky')
        se_b = aux.setrunc.run(self.se, self.fock, nmom, method='band', qr='cholesky')
        se_c = aux.setrunc.run(self.se, self.fock, nmom, method='block', qr='numpy')
        se_d = aux.setrunc.run(self.se, self.fock, nmom, method='band', qr='scipy')

        m = self.se.moment(range(2*(nmom+1)))
        m_a = se_a.moment(range(2*(nmom+1)))
        m_b = se_b.moment(range(2*(nmom+1)))
        m_c = se_c.moment(range(2*(nmom+1)))
        m_d = se_d.moment(range(2*(nmom+1)))
        for n in range(2*(nmom+1)):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_b[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_c[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_d[n])), 0, 12-2*n)

        m = self.se.as_occupied().moment(range(2*(nmom+1)))
        m_a = se_a.as_occupied().moment(range(2*(nmom+1)))
        m_b = se_b.as_occupied().moment(range(2*(nmom+1)))
        m_c = se_c.as_occupied().moment(range(2*(nmom+1)))
        m_d = se_d.as_occupied().moment(range(2*(nmom+1)))
        for n in range(2*(nmom+1)):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_b[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_c[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_d[n])), 0, 12-2*n)

        m = self.se.as_virtual().moment(range(2*(nmom+1)))
        m_a = se_a.as_virtual().moment(range(2*(nmom+1)))
        m_b = se_b.as_virtual().moment(range(2*(nmom+1)))
        m_c = se_c.as_virtual().moment(range(2*(nmom+1)))
        m_d = se_d.as_virtual().moment(range(2*(nmom+1)))
        for n in range(2*(nmom+1)):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_b[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_c[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_d[n])), 0, 12-2*n)

    def test_setrunc_1(self):
        nmom = 1
        se_a = aux.setrunc.run(self.se, self.fock, nmom, method='block', qr='cholesky')
        se_b = aux.setrunc.run(self.se, self.fock, nmom, method='band', qr='cholesky')
        se_c = aux.setrunc.run(self.se, self.fock, nmom, method='block', qr='numpy')
        se_d = aux.setrunc.run(self.se, self.fock, nmom, method='band', qr='scipy')

        m = self.se.moment(range(2*(nmom+1)))
        m_a = se_a.moment(range(2*(nmom+1)))
        m_b = se_b.moment(range(2*(nmom+1)))
        m_c = se_c.moment(range(2*(nmom+1)))
        m_d = se_d.moment(range(2*(nmom+1)))
        for n in range(2*(nmom+1)):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_b[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_c[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_d[n])), 0, 12-2*n)

        m = self.se.as_occupied().moment(range(2*(nmom+1)))
        m_a = se_a.as_occupied().moment(range(2*(nmom+1)))
        m_b = se_b.as_occupied().moment(range(2*(nmom+1)))
        m_c = se_c.as_occupied().moment(range(2*(nmom+1)))
        m_d = se_d.as_occupied().moment(range(2*(nmom+1)))
        for n in range(2*(nmom+1)):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_b[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_c[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_d[n])), 0, 12-2*n)

        m = self.se.as_virtual().moment(range(2*(nmom+1)))
        m_a = se_a.as_virtual().moment(range(2*(nmom+1)))
        m_b = se_b.as_virtual().moment(range(2*(nmom+1)))
        m_c = se_c.as_virtual().moment(range(2*(nmom+1)))
        m_d = se_d.as_virtual().moment(range(2*(nmom+1)))
        for n in range(2*(nmom+1)):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_b[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_c[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_d[n])), 0, 12-2*n)

    def test_setrunc_5(self):
        nmom = 5
        se_a = aux.setrunc.run(self.se, self.fock, nmom, method='block', qr='cholesky')
        se_b = aux.setrunc.run(self.se, self.fock, nmom, method='band', qr='cholesky')
        se_c = aux.setrunc.run(self.se, self.fock, nmom, method='block', qr='numpy')
        se_d = aux.setrunc.run(self.se, self.fock, nmom, method='band', qr='scipy')

        m = self.se.moment(range(2*(nmom+1)))
        m_a = se_a.moment(range(2*(nmom+1)))
        m_b = se_b.moment(range(2*(nmom+1)))
        m_c = se_c.moment(range(2*(nmom+1)))
        m_d = se_d.moment(range(2*(nmom+1)))
        for n in range(2*(nmom+1)):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_b[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_c[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_d[n])), 0, 12-2*n)

        m = self.se.as_occupied().moment(range(2*(nmom+1)+1))
        m_a = se_a.as_occupied().moment(range(2*(nmom+1)+1))
        m_b = se_b.as_occupied().moment(range(2*(nmom+1)+1))
        m_c = se_c.as_occupied().moment(range(2*(nmom+1)+1))
        m_d = se_d.as_occupied().moment(range(2*(nmom+1)+1))
        for n in range(2*(nmom+1)):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_b[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_c[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_d[n])), 0, 12-2*n)

        m = self.se.as_virtual().moment(range(2*(nmom+1)+1))
        m_a = se_a.as_virtual().moment(range(2*(nmom+1)+1))
        m_b = se_b.as_virtual().moment(range(2*(nmom+1)+1))
        m_c = se_c.as_virtual().moment(range(2*(nmom+1)+1))
        m_d = se_d.as_virtual().moment(range(2*(nmom+1)+1))
        for n in range(2*(nmom+1)):
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_a[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_b[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_c[n])), 0, 12-2*n)
            self.assertAlmostEqual(np.max(np.absolute(m[n] - m_d[n])), 0, 12-2*n)


if __name__ == '__main__':
    unittest.main()
