import unittest
import numpy as np

from auxgf import mol, hf, adc


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')
        self.rhf = hf.RHF(self.m).run(conv_tol=1e-14)
        self.uhf = hf.UHF(self.m).run(conv_tol=1e-14)

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.uhf

    def test_radc2_ecorr(self):
        adc2 = adc.ADC2(self.rhf).run()
        self.assertAlmostEqual(adc2.e_corr, -0.20905684838137956, 8)

    def test_radc2_ip(self):
        adc2 = adc.ADC2(self.rhf).run()
        self.assertAlmostEqual(adc2.ip[0], 0.398405767574538, 8)

    def test_radc2_ea(self):
        adc2 = adc.ADC2(self.rhf).run()
        self.assertAlmostEqual(adc2.ea[0], 0.15307444259968273, 8)

    def test_uadc2_ecorr(self):
        adc2 = adc.ADC2(self.uhf).run()
        self.assertAlmostEqual(adc2.e_corr, -0.20905684837248664, 8)

    def test_uadc2_ip(self):
        adc2 = adc.ADC2(self.uhf).run()
        self.assertAlmostEqual(adc2.ip[0], 0.3984057677344674, 8)

    def test_uadc2_ea(self):
        adc2 = adc.ADC2(self.uhf).run()
        self.assertAlmostEqual(adc2.ea[0], 0.15307444260447728, 8)


if __name__ == '__main__':
    unittest.main()
