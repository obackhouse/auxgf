import unittest
import numpy as np

from auxgf import mol, hf


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')
        self.rhf = hf.RHF(self.m).run()
        self.rhf_df = hf.RHF(self.m, with_df=True).run()

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.rhf_df

    def test_e_tot(self):
        e1 = self.rhf.e_tot
        e2 = self.rhf.energy_1body(self.rhf.h1e_ao, self.rhf.rdm1_ao, self.rhf.fock_ao) + self.rhf.e_nuc 
        e3 = self.rhf.energy_1body(self.rhf.h1e_mo, self.rhf.rdm1_mo, self.rhf.fock_mo) + self.rhf.e_nuc
        self.assertAlmostEqual(e1, -76.01678947206919, 8)
        self.assertAlmostEqual(e2, -76.01678947206919, 8)
        self.assertAlmostEqual(e3, -76.01678947206919, 8)

    def test_e(self):
        self.assertAlmostEqual(self.rhf.e[0], -20.565311186745017, 8)

    def test_chempot(self):
        self.assertAlmostEqual(self.rhf.chempot, -0.16055580940894087, 8)
        
    def test_rdm1(self):
        nelec = np.trace(self.rhf.rdm1_mo)
        self.assertAlmostEqual(nelec, self.m.nelec, 8)
        w, v = np.linalg.eigh(self.rhf.rdm1_mo)
        self.assertAlmostEqual(np.max(np.absolute(w[:self.rhf.nvir] - 0)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(w[self.rhf.nvir:] - 2)), 0, 8)

    def test_converged(self):
        self.assertTrue(self.rhf._pyscf.converged)

    def test_fock_ao(self):
        f1 = self.rhf.get_fock(self.rhf.rdm1_ao, basis='ao')
        self.assertAlmostEqual(np.max(np.absolute(f1 - self.rhf.fock_ao)), 0, 8)

    def test_fock_mo(self):
        f1 = self.rhf.get_fock(self.rhf.rdm1_mo, basis='mo')
        self.assertAlmostEqual(np.max(np.absolute(f1 - self.rhf.fock_mo)), 0, 8)

    def test_density_fit(self):
        self.assertTrue(self.rhf_df.eri_ao.ndim == 3)
        self.assertTrue(self.rhf_df.eri_mo.ndim == 3)

    def test_from_pyscf(self):
        self.rhf_fp = self.rhf.from_pyscf(self.rhf._pyscf)
        self.rhf_df_fp = self.rhf_df.from_pyscf(self.rhf_df._pyscf)
        self.assertAlmostEqual(self.rhf.e_tot, self.rhf_fp.e_tot, 10)
        self.assertAlmostEqual(self.rhf_df.e_tot, self.rhf_df_fp.e_tot, 10)
        

if __name__ == '__main__':
    unittest.main()
