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

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf

    def test_radc2_ecorr(self):
        adc2 = adc.RADC2(self.rhf, verbose=False).run()
        self.assertAlmostEqual(adc2.e_corr, -0.20905684836546554, 8)

    def test_radc2_ip(self):
        adc2 = adc.RADC2(self.rhf, method='ip', verbose=False).run()
        self.assertAlmostEqual(adc2.ip[0], 0.3984057676983328, 8)

    def test_radc2_ea(self):
        adc2 = adc.RADC2(self.rhf, method='ea', verbose=False).run()
        self.assertAlmostEqual(adc2.ea[0], 0.15307444258591507, 8)


if __name__ == '__main__':
    unittest.main()
