import unittest
import numpy as np

from auxgf import mol, hf, adc


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')
        self.rhf = hf.RHF(self.m).run()
        self.e_mp2 = -0.20905684700662164
        self.e_mp2_scs = -0.20503556854447708
        self.ip = 0.39840678852582134
        self.ea = 0.15307442008807248

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.e_mp2, self.e_mp2_scs

    def test_radc2_ecorr(self):
        adc2 = adc.RADC2(self.rhf, verbose=False).run()
        self.assertAlmostEqual(adc2.e_corr, self.e_mp2, 8)

    def test_radc2_ip(self):
        adc2 = adc.RADC2(self.rhf, method='ip', verbose=False).run()
        self.assertAlmostEqual(adc2.ip[0], self.ip, 8)

    def test_radc2_ea(self):
        adc2 = adc.RADC2(self.rhf, method='ea', verbose=False).run()
        self.assertAlmostEqual(adc2.ea[0], self.ea, 8)

    def test_radc2_truncated(self):
        adc2 = adc.RADC2(self.rhf, verbose=False, nmom=(5,5)).run()
        self.assertAlmostEqual(adc2.e_corr, self.e_mp2, 3)
        adc2 = adc.RADC2(self.rhf, verbose=False, nmom=(2,2)).run()
        self.assertAlmostEqual(adc2.e_corr, self.e_mp2, 2)

    def test_scs_radc2(self):
        adc2 = adc.RADC2(self.rhf, verbose=False, os_factor=1.2, ss_factor=0.33).run()
        self.assertAlmostEqual(adc2.e_corr, self.e_mp2_scs, 8)


if __name__ == '__main__':
    unittest.main()
