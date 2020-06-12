import unittest
import numpy as np

from auxgf import util

np.random.seed(1)


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)

    @classmethod
    def tearDownClass(self):
        pass

    def test_einsum_dot(self):
        ab = np.random.random((5, 6))
        bc = np.random.random((6, 7))
        m = np.dot(ab, bc)
        m1 = util.linalg.numpy_einsum('ab,bc->ac', ab, bc)
        m2 = util.linalg.pyscf_einsum('ab,bc->ac', ab, bc)
        m3 = util.linalg._tblis_einsum('ab,bc->ac', ab, bc)
        self.assertAlmostEqual(np.max(np.absolute(m - m1)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(m - m2)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(m - m3)), 0, 8)

    def test_einsum_tensorsum(self):
        abcd = np.random.random((5, 6, 7, 8))
        cbda = np.random.random((7, 6, 8, 5))
        m = np.sum(abcd * cbda.transpose(3,1,0,2))
        m1 = util.linalg.numpy_einsum('abcd,cbda->', abcd, cbda)
        m2 = util.linalg.pyscf_einsum('abcd,cbda->', abcd, cbda)
        m3 = util.linalg._tblis_einsum('abcd,cbda->', abcd, cbda)
        self.assertAlmostEqual(np.max(np.absolute(m - m1)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(m - m2)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(m - m3)), 0, 8)

    def test_dgemm_cc(self):
        a = np.array(np.random.random((5,6)), order='C')
        b = np.array(np.random.random((6,7)), order='C')
        m1 = np.dot(a, b)
        m2 = util.linalg.dgemm(a, b)
        self.assertTrue(util.linalg._is_contiguous(a))
        self.assertTrue(util.linalg._is_contiguous(b))
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 8)

    def test_dgemm_cf(self):
        a = np.array(np.random.random((5,6)), order='C')
        b = np.array(np.random.random((6,7)), order='F')
        m1 = np.dot(a, b)
        m2 = util.linalg.dgemm(a, b)
        self.assertTrue(util.linalg._is_contiguous(a))
        self.assertTrue(util.linalg._is_contiguous(b))
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 8)

    def test_dgemm_fc(self):
        a = np.array(np.random.random((5,6)), order='F')
        b = np.array(np.random.random((6,7)), order='C')
        m1 = np.dot(a, b)
        m2 = util.linalg.dgemm(a, b)
        self.assertTrue(util.linalg._is_contiguous(a))
        self.assertTrue(util.linalg._is_contiguous(b))
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 8)

    def test_dgemm_ff(self):
        a = np.array(np.random.random((5,6)), order='F')
        b = np.array(np.random.random((6,7)), order='F')
        m1 = np.dot(a, b)
        m2 = util.linalg.dgemm(a, b)
        self.assertTrue(util.linalg._is_contiguous(a))
        self.assertTrue(util.linalg._is_contiguous(b))
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 8)

    def test_dgemm(self):
        a = np.array(np.random.random((5,6)), order='C')
        b = np.array(np.random.random((6,7)), order='C')
        c = np.array(np.random.random((5,7)), order='C')
        alpha = 0.5
        beta = 0.25
        m1 = beta * c + alpha * np.dot(a, b)
        m2 = util.linalg.dgemm(a, b, c=c, alpha=alpha, beta=beta)
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 8)

    def test_cholesky_qr(self):
        a = np.random.random((50, 50))
        a = np.dot(a, a.T)
        q1, r1 = util.linalg.cholesky_qr(a)
        q2, r2 = util.linalg.cholesky_qr2(a)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(q1, r1) - a)), 0, 6)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(q2, r2) - a)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(q1.T, q1) - np.eye(50))), 0, 4)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(q2.T, q2) - np.eye(50))), 0, 8)

    def test_block_diag(self):
        b1 = np.ones((2, 2))
        b2 = np.ones((2, 3)) * 2
        b3 = np.ones((1, 4)) * 3
        m1 = np.zeros((5, 9))
        m1[:2,:2] = 1
        m1[2:4,2:5] = 2
        m1[4:5,5:9] = 3
        m2 = util.linalg.block_diag([b1, b2, b3])
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 12)

    def test_diagonal(self):
        a = np.zeros((10, 10))
        diag = util.linalg.diagonal(a)
        diag[:] = np.ones((10,))
        self.assertAlmostEqual(np.max(np.absolute(a - np.eye(10))), 0, 12)

    def test_spin_block_1d(self):
        a1 = np.random.random((5))
        a2 = np.random.random((5))
        m1 = util.linalg.spin_block(a1, a2)
        m2 = np.concatenate((a1, a2), axis=0)
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 12)

    def test_spin_block_2d(self):
        a1 = np.random.random((5, 5))
        a2 = np.random.random((5, 5))
        m1 = util.linalg.spin_block(a1, a2)
        m2 = np.block([[a1, np.zeros_like(a1)], [np.zeros_like(a1), a2]])
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 12)

    def test_spin_block_4d(self):
        a1 = np.random.random((5, 5, 5, 5))
        a2 = np.random.random((5, 5, 5, 5))
        m1 = util.linalg.spin_block(a1, a2)
        m2 = np.kron(np.eye(2), np.kron(np.eye(2), a1).T)
        m2[tuple([slice(x, None) for x in a1.shape])] = a2
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 12)

    def test_is_hermitian(self):
        a = np.random.random((10, 10))
        b = a.copy() + a.T.copy()
        c = b.astype(np.complex128)
        c[np.tril_indices_from(c)] += 1.0j * b[np.tril_indices_from(c)]
        c[np.triu_indices_from(c)] += -1.0j * b[np.triu_indices_from(c)]
        self.assertFalse(util.linalg.is_hermitian(a))
        self.assertTrue(util.linalg.is_hermitian(b))
        self.assertTrue(util.linalg.is_hermitian(c))

    def test_outer_sum(self):
        a = np.random.random((5))
        b = np.random.random((5))
        c = np.random.random((5))
        m1 = a[:,None,None] + b[None,:,None] + c[None,None,:]
        m2 = util.linalg.outer_sum([a, b, c])
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 12)

    def test_outer_mul(self):
        a = np.random.random((5))
        b = np.random.random((5))
        c = np.random.random((5))
        m1 = a[:,None,None] * b[None,:,None] * c[None,None,:]
        m2 = util.linalg.outer_mul([a, b, c])
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 12)

    def test_normalise(self):
        a = np.random.random((5, 5))
        a = util.linalg.normalise(a)
        lengths = np.sqrt(np.sum(a*a, axis=0))
        self.assertAlmostEqual(np.max(np.absolute(lengths - 1)), 0, 10)

    def test_batch_eigh(self):
        mats =  [np.random.random((3,3)) for x in range(4)]
        mats += [np.random.random((4,4)) for x in range(3)]
        mats += [np.random.random((5,5)) for x in range(2)]
        mats = [np.dot(x, x.T) for x in mats]
        ws, vs = util.linalg.batch_eigh(mats)
        for i in range(len(mats)):
            w, v = ws[i], vs[i]
            m = np.dot(np.dot(v, np.diag(w)), np.linalg.inv(v))
            self.assertAlmostEqual(np.max(np.absolute(m - mats[i])), 0, 8)


if __name__ == '__main__':
    unittest.main()
