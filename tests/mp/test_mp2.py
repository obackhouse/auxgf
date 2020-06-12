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
        self.uhf = hf.UHF(self.m).run()
        self.e_rmp2 = -0.20905684700662164
        self.e_ump2 = -0.20905685057662993

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.uhf, self.e_rmp2, self.e_ump2

    def test_rmp2(self):
        mp2 = mp.MP2(self.rhf).run()
        self.assertAlmostEqual(mp2.e_corr, self.e_rmp2, 8)

    def test_ump2(self):
        mp2 = mp.MP2(self.uhf).run()
        self.assertAlmostEqual(mp2.e_corr, self.e_ump2, 8)


if __name__ == '__main__':
    unittest.main()
