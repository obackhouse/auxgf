import unittest
import numpy as np

from auxgf import mol, hf


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1)
        self.uhf = hf.UHF(self.m).run()
        self.uhf_df = hf.UHF(self.m, with_df=True).run()

    @classmethod
    def tearDownClass(self):
        del self.m, self.uhf, self.uhf_df

    def test_e_tot(self):
        e1 = self.uhf.e_tot
        e2 = self.uhf.energy_1body(self.uhf.h1e_ao, self.uhf.rdm1_ao, self.uhf.fock_ao) + self.uhf.e_nuc 
        e3 = self.uhf.energy_1body(self.uhf.h1e_mo, self.uhf.rdm1_mo, self.uhf.fock_mo) + self.uhf.e_nuc
        self.assertAlmostEqual(e1, -75.39232283693659, 8)
        self.assertAlmostEqual(e2, -75.39232283693659, 8)
        self.assertAlmostEqual(e3, -75.39232283693659, 8)

    def test_e(self):
        self.assertAlmostEqual(self.uhf.e[0][0], -20.6287416890306, 8)
        self.assertAlmostEqual(self.uhf.e[1][0], -20.58850727764275, 8)

    def test_chempot(self):
        self.assertAlmostEqual(self.uhf.chempot[0], -0.18219091973523854, 8)
        self.assertAlmostEqual(self.uhf.chempot[1], -0.17948442207826365, 8) 
        
    def test_rdm1(self):
        nalph = np.trace(self.uhf.rdm1_mo[0])
        nbeta = np.trace(self.uhf.rdm1_mo[1])
        self.assertAlmostEqual(nalph, self.m.nalph, 8)
        self.assertAlmostEqual(nbeta, self.m.nbeta, 8)
        w, v = np.linalg.eigh(self.uhf.rdm1_mo)
        self.assertAlmostEqual(np.max(np.absolute(w[:,:self.uhf.nvir[0]] - 0)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(w[:,self.uhf.nvir[1]:] - 1)), 0, 8)

    def test_converged(self):
        self.assertTrue(self.uhf._pyscf.converged)

    def test_fock_ao(self):
        f1 = self.uhf.get_fock(self.uhf.rdm1_ao, basis='ao')
        self.assertAlmostEqual(np.max(np.absolute(f1 - self.uhf.fock_ao)), 0, 8)

    def test_fock_mo(self):
        f1 = self.uhf.get_fock(self.uhf.rdm1_mo, basis='mo')
        self.assertAlmostEqual(np.max(np.absolute(f1 - self.uhf.fock_mo)), 0, 8)

    def test_density_fit(self):
        self.assertTrue(self.uhf_df.eri_ao.ndim == 3)
        self.assertTrue(self.uhf_df.eri_mo.ndim == 4)

    def test_from_pyscf(self):
        self.uhf_fp = self.uhf.from_pyscf(self.uhf._pyscf)
        self.uhf_df_fp = self.uhf_df.from_pyscf(self.uhf_df._pyscf)
        self.assertAlmostEqual(self.uhf.e_tot, self.uhf_fp.e_tot, 10)
        self.assertAlmostEqual(self.uhf_df.e_tot, self.uhf_df_fp.e_tot, 10)
        

if __name__ == '__main__':
    unittest.main()
