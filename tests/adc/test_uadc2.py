import unittest
import numpy as np

from auxgf import mol, hf, adc


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1)
        self.uhf = hf.UHF(self.m).run()
        self.e_mp2 = -0.15197757655845123
        self.e_mp2_scs = -0.1502359064727459
        self.ip = 0.4651687192343633
        self.ea = 0.005690350422160151

    @classmethod
    def tearDownClass(self):
        del self.m, self.uhf

    def test_uadc2_ecorr(self):
        adc2 = adc.UADC2(self.uhf, verbose=False).run()
        self.assertAlmostEqual(adc2.e_corr, self.e_mp2, 8)

    def test_uadc2_ip(self):
        adc2 = adc.UADC2(self.uhf, method='ip', verbose=False).run()
        self.assertAlmostEqual(adc2.ip[0], self.ip, 8)

    def test_uadc2_ea(self):
        adc2 = adc.UADC2(self.uhf, method='ea', verbose=False).run()
        self.assertAlmostEqual(adc2.ea[0], self.ea, 8)

    def test_uadc2_truncated(self):
        adc2 = adc.UADC2(self.uhf, verbose=False, nmom=(5,5)).run()
        self.assertAlmostEqual(adc2.e_corr, self.e_mp2, 3)
        adc2 = adc.UADC2(self.uhf, verbose=False, nmom=(2,2)).run()
        self.assertAlmostEqual(adc2.e_corr, self.e_mp2, 2)


if __name__ == '__main__':
    unittest.main()
