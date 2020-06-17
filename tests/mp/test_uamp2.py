import unittest
import numpy as np

from auxgf import mol, hf, mp


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1)
        self.uhf = hf.UHF(self.m).run()
        self.e_mp2 = -0.15197757655845123
        self.e_mp2_scs = -0.1502359064727459

    @classmethod
    def tearDownClass(self):
        del self.m, self.uhf, self.e_mp2, self.e_mp2_scs

    def test_uamp2(self):
        uamp2 = mp.UAMP2(self.uhf, verbose=False).run()
        self.assertAlmostEqual(uamp2.e_corr, self.e_mp2, 8)

    def test_uamp2_truncated(self):
        uamp2 = mp.UAMP2(self.uhf, verbose=False, nmom=(5,5)).run()
        self.assertAlmostEqual(uamp2.e_corr, self.e_mp2, 3)
        uamp2 = mp.UAMP2(self.uhf, verbose=False, nmom=(2,2)).run()
        self.assertAlmostEqual(uamp2.e_corr, self.e_mp2, 2)

    def test_scs_uamp2(self):
        uamp2 = mp.UAMP2(self.uhf, verbose=False, os_factor=1.2, ss_factor=0.33).run()
        self.assertAlmostEqual(uamp2.e_corr, self.e_mp2_scs)


if __name__ == '__main__':
    unittest.main()
