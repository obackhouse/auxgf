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

    def test_radc2x_ecorr(self):
        adc2 = adc.ADC2x(self.rhf).run()
        self.assertAlmostEqual(adc2.e_corr, -0.20905684838137956, 8)

    def test_radc2x_ip(self):
        adc2 = adc.ADC2x(self.rhf).run()
        self.assertAlmostEqual(np.ravel(adc2.ip[0])[0], 0.40478204113376104, 8)
                                                      
    def test_radc2x_ea(self):                         
        adc2 = adc.ADC2x(self.rhf).run()
        self.assertAlmostEqual(np.ravel(adc2.ea[0])[0], 0.14962281081945084, 8)

    def test_uadc2x_ecorr(self):
        adc2 = adc.ADC2x(self.uhf).run()
        self.assertAlmostEqual(adc2.e_corr, -0.20905684838137956, 8)

    def test_uadc2x_ip(self):
        adc2 = adc.ADC2x(self.uhf).run()
        self.assertAlmostEqual(np.ravel(adc2.ip[0])[0], 0.40478204113376104, 8)
                                                      
    def test_uadc2x_ea(self):                         
        adc2 = adc.ADC2x(self.uhf).run()
        self.assertAlmostEqual(np.ravel(adc2.ea[0])[0], 0.14962281081945084, 8)


if __name__ == '__main__':
    unittest.main()
