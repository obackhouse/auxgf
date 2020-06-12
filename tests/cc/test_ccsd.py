import unittest
import numpy as np

from auxgf import mol, hf, cc


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')
        self.rhf = hf.RHF(self.m).run()

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf

    def test_ccsd_ecorr(self):
        ccsd = cc.CCSD(self.rhf).run()
        self.assertAlmostEqual(ccsd.e_corr, -0.21807391034114687, 8)

    def test_ccsd_ip(self):
        ccsd = cc.CCSD(self.rhf).run()
        self.assertAlmostEqual(ccsd.ip[0], 0.4317431075627711, 8)

    def test_ccsd_ip(self):
        ccsd = cc.CCSD(self.rhf).run()
        self.assertAlmostEqual(ccsd.ea[0], 0.1554583211994489, 8)


if __name__ == '__main__':
    unittest.main()
