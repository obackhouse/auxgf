import unittest
import numpy as np

from auxgf import mol, hf, mp


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')
        self.rhf = hf.RHF(self.m).run()
        self.e_mp2 = -0.20905684700662164
        self.e_mp2_scs = -0.20503556854447708

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.e_mp2, self.e_mp2_scs

    def test_ramp2(self):
        ramp2 = mp.RAMP2(self.rhf, verbose=False).run()
        self.assertAlmostEqual(ramp2.e_corr, self.e_mp2, 8)

    def test_ramp2_truncated(self):
        ramp2 = mp.RAMP2(self.rhf, verbose=False, nmom=(5,5)).run()
        self.assertAlmostEqual(ramp2.e_corr, self.e_mp2, 3)
        ramp2 = mp.RAMP2(self.rhf, verbose=False, nmom=(2,2)).run()
        self.assertAlmostEqual(ramp2.e_corr, self.e_mp2, 2)

    def test_scs_ramp2(self):
        ramp2 = mp.RAMP2(self.rhf, verbose=False, os_factor=1.2, ss_factor=0.33).run()
        self.assertAlmostEqual(ramp2.e_corr, self.e_mp2_scs)


if __name__ == '__main__':
    unittest.main()
