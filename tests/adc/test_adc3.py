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

    def test_radc3_ecorr(self):
        adc2 = adc.ADC3(self.rhf).run()
        self.assertAlmostEqual(adc2.e_corr, -0.21508728065561694, 8)

    def test_radc3_ip(self):
        adc2 = adc.ADC3(self.rhf).run()
        self.assertAlmostEqual(np.ravel(adc2.ip[0])[0], 0.44743837873571013, 8)
                                                      
    def test_radc3_ea(self):                          
        adc2 = adc.ADC3(self.rhf).run()
        self.assertAlmostEqual(np.ravel(adc2.ea[0])[0], 0.1547289617248287, 8)

    def test_uadc3_ecorr(self):
        adc2 = adc.ADC3(self.uhf).run()
        self.assertAlmostEqual(adc2.e_corr, -0.2150872806327277, 8)

    def test_uadc3_ip(self):
        adc2 = adc.ADC3(self.uhf).run()
        self.assertAlmostEqual(np.ravel(adc2.ip[0])[0], 0.4474383788021712, 8)
                                                      
    def test_uadc3_ea(self):                          
        adc2 = adc.ADC3(self.uhf).run()
        self.assertAlmostEqual(np.ravel(adc2.ea[0])[0], 0.15472896174165018, 8)


if __name__ == '__main__':
    unittest.main()
