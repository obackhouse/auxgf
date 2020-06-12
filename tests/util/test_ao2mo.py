import unittest
import numpy as np

from auxgf import util


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)

    @classmethod
    def tearDownClass(self):
        pass

    def test_2d_rr(self):
        m = np.random.random((5, 5))
        c1 = np.random.random((5, 10))
        c2 = np.random.random((5, 10))
        t1 = util.ao2mo(m, c1, c2)
        t2 = util.einsum('pq,pi,qj->ij', m, c1, c2)
        self.assertAlmostEqual(np.max(np.absolute(t1 - t2)), 0, 8)

    def test_2d_ru(self):
        m = np.random.random((5, 5))
        c1 = np.random.random((2, 5, 10))
        c2 = np.random.random((2, 5, 10))
        t1 = util.ao2mo(m, c1, c2)
        t2 = util.einsum('pq,spi,sqj->sij', m, c1, c2)
        self.assertAlmostEqual(np.max(np.absolute(t1 - t2)), 0, 8)

    def test_2d_uu(self):
        m = np.random.random((2, 5, 5))
        c1 = np.random.random((2, 5, 10))
        c2 = np.random.random((2, 5, 10))
        t1 = util.ao2mo(m, c1, c2)
        t2 = util.einsum('spq,spi,sqj->sij', m, c1, c2)
        self.assertAlmostEqual(np.max(np.absolute(t1 - t2)), 0, 8)

    def test_4d_rr(self):
        m = np.random.random((5, 5, 5, 5))
        c1 = np.random.random((5, 10))
        c2 = np.random.random((5, 10))
        c3 = np.random.random((5, 10))
        c4 = np.random.random((5, 10))
        t1 = util.ao2mo(m, c1, c2, c3, c4)
        t2 = util.einsum('pqrs,pi,qj,rk,sl->ijkl', m, c1, c2, c3, c4)
        self.assertAlmostEqual(np.max(np.absolute(t1 - t2)), 0, 8)

    def test_4d_ru(self):
        m = np.random.random((5, 5, 5, 5))
        c1 = np.random.random((2, 5, 10))
        c2 = np.random.random((2, 5, 10))
        c3 = np.random.random((2, 5, 10))
        c4 = np.random.random((2, 5, 10))
        t1 = util.ao2mo(m, c1, c2, c3, c4)
        t2 = util.einsum('pqrs,api,aqj,brk,bsl->abijkl', m, c1, c2, c3, c4)
        self.assertAlmostEqual(np.max(np.absolute(t1 - t2)), 0, 8)

    def test_4d_uu(self):
        m = np.random.random((2, 2, 5, 5, 5, 5))
        c1 = np.random.random((2, 5, 10))
        c2 = np.random.random((2, 5, 10))
        c3 = np.random.random((2, 5, 10))
        c4 = np.random.random((2, 5, 10))
        t1 = util.ao2mo(m, c1, c2, c3, c4)
        t2 = util.einsum('abpqrs,api,aqj,brk,bsl->abijkl', m, c1, c2, c3, c4)
        self.assertAlmostEqual(np.max(np.absolute(t1 - t2)), 0, 8)

    def test_mo2qo(self):
        m = np.random.random((5, 5, 5, 5))
        c1 = np.random.random((5, 10))
        c2 = np.random.random((5, 10))
        c3 = np.random.random((5, 10))
        t1 = util.mo2qo(m, c1, c2, c3)
        t2 = util.einsum('pqrs,qi,rj,sk->pijk', m, c1, c2, c3)
        self.assertAlmostEqual(np.max(np.absolute(t1 - t2)), 0, 8)


if __name__ == '__main__':
    unittest.main()
