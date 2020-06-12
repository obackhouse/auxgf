import unittest
import numpy as np
import os

from auxgf import util, mol, hf, aux, grids


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g')
        self.rhf = hf.RHF(self.m).run()
        self.fock = self.rhf.fock_mo
        self.se = aux.build_rmp2(self.rhf.e, self.rhf.eri_mo, chempot=self.rhf.chempot) #FIXME remove test dependency?
        self.imfq = grids.ImFqGrid(2**5, beta=2**3)

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.fock, self.se, self.imfq

    def test_build_denominator(self):
        se = self.se
        d1 = 1.0 / (1.0j * self.imfq[:,None] - se.e + se.chempot)
        d2 = se.build_denominator(self.imfq, se.e, se.v, chempot=se.chempot)
        self.assertAlmostEqual(np.max(np.absolute(d1 - d2)), 0, 8)

    def test_as_spectrum(self):
        se = self.se
        f1 = util.einsum('xk,yk,wk->wxy', se.v, se.v, se.build_denominator(self.imfq, se.e, se.v, chempot=se.chempot))
        f2 = se.as_spectrum(self.imfq)
        self.assertAlmostEqual(np.max(np.absolute(f1 - f2)), 0, 8)

    def test_build_derivative(self):
        se = self.se
        df1 = -util.einsum('xk,yk,wk->wxy', se.v, se.v, se.build_denominator(self.imfq, se.e, se.v, chempot=se.chempot)**2)
        df2 = se.build_derivative(self.imfq, se.e, se.v, chempot=se.chempot)
        self.assertAlmostEqual(np.max(np.absolute(df1 - df2)), 0, 8)

    def test_as_hamiltonian(self):
        se = self.se
        f1 = np.block([[self.fock, se.v], [se.v.T, np.diag(se.e)]])
        f2 = se.as_hamiltonian(self.fock)
        self.assertAlmostEqual(np.max(np.absolute(f1 - f2)), 0, 8)

    def test_as_occupied(self):
        se = self.se
        occ = se.as_occupied()
        self.assertEqual(occ.naux, se.nocc)
        self.assertTrue(np.all(occ.e < se.chempot))

    def test_as_virtual(self):
        se = self.se
        vir = se.as_virtual()
        self.assertEqual(vir.naux, se.nvir)
        self.assertTrue(np.all(vir.e >= se.chempot))

    def test_as_window(self):
        se = self.se
        win = se.as_window(-1.5, 1.5)
        self.assertTrue(np.all(win.e < 1.5) and np.all(win.e > -1.5))

    def test_dot(self):
        se = self.se
        f_ext = se.as_hamiltonian(self.fock)
        vec = np.random.random((f_ext.shape[0]))
        m1 = np.dot(f_ext, vec)
        m2 = se.dot(self.fock, vec)
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 8)

    def test_eig(self):
        se = self.se
        f_ext = se.as_hamiltonian(self.fock)
        w1, v1 = np.linalg.eigh(f_ext)
        w2, v2 = se.eig(self.fock)
        self.assertAlmostEqual(np.max(np.absolute(w1 - w2)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(v1 - v2)), 0, 8)

    def test_moment(self):
        se = self.se
        m1 = util.einsum('xk,nk,yk->nxy', se.v, se.e[None,:] ** np.arange(4)[:,None], se.v)
        m2 = se.moment(range(4))
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 8)

    #def test_merge(self):
    #    #FIXME
    #    pass

    def test_se_compress(self):
        se = self.se
        se_red = se.se_compress(self.fock, 3)
        m1 = se.moment(range(5))
        m2 = se_red.moment(range(5))
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 8)

    def test_gf_compress(self):
        se = self.se
        se_red = se.gf_compress(self.fock, 3)
        m1 = se.moment(range(5))
        m2 = se_red.moment(range(5))
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 8)

    def test_compress(self):
        se = self.se
        se_red = se.compress(self.fock, nmom=(3, 3))
        m1 = se.moment(range(5))
        m2 = se_red.moment(range(5))
        self.assertAlmostEqual(np.max(np.absolute(m1 - m2)), 0, 8)

    #def test_fit(self):
    #    #FIXME
    #    pass

    def test_copy(self):
        se = self.se
        se_copy = se.copy()
        self.assertAlmostEqual(np.max(np.absolute(se.e - se_copy.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(se.v - se_copy.v)), 0, 12)
        self.assertFalse(np.shares_memory(se.e, se_copy.e))
        self.assertFalse(np.shares_memory(se.v, se_copy.v))

    def test_e_sort(self):
        se = self.se
        se_sort = se.copy()
        se_sort.sort(which='e')
        self.assertAlmostEqual(np.max(np.absolute(np.sort(se.e) - se_sort.e)), 0, 12)

    def test_w_sort(self):
        se = self.se
        se_sort = se.copy()
        se_sort.sort(which='w')
        self.assertAlmostEqual(np.max(np.absolute(np.sort(se.w) - se_sort.w)), 0, 12)

    def test_save_load(self):
        se = self.se
        se.save('tmp')
        se_load = se.load('tmp')
        os.remove('tmp')
        self.assertAlmostEqual(np.max(np.absolute(se.e - se_load.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(se.v - se_load.v)), 0, 12)

    def test_memsize(self):
        se = self.se
        nbytes = se.e.nbytes + se.v.nbytes + 8
        gbs = nbytes / 1e9
        self.assertAlmostEqual(abs(gbs - se.memsize()), 0, 8)

    def test_new(self):
        se = self.se
        se_new = se.new(np.random.random(se.e.shape), np.random.random(se.v.shape))
        self.assertEqual(se.chempot, se_new.chempot)
        self.assertEqual(se.nphys, se_new.nphys)
        self.assertEqual(se.naux, se_new.naux)

    def test_add(self):
        se = self.se
        se_add = se.as_occupied() + se.as_virtual()
        self.assertAlmostEqual(np.max(np.absolute(se.e - se_add.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(se.v - se_add.v)), 0, 12)

    def test_equals(self):
        se = self.se
        se_copy = se.copy()
        self.assertTrue(se == se_copy)


if __name__ == '__main__':
    unittest.main()
