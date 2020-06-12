import unittest
import numpy as np

from auxgf import mol, hf, mp

# not very rigorous


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g')
        self.rhf = hf.RHF(self.m).run()
        self.uhf = hf.UHF(self.m).run()
        self.e_rmp2 = mp.MP2(self.rhf).run().e_corr
        self.e_ump2 = mp.MP2(self.uhf).run().e_corr

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.uhf, self.e_rmp2, self.e_ump2

    def test_roomp2(self):
        mp2 = mp.OOMP2(self.rhf).run()
        self.assertAlmostEqual(mp2.e_corr, self.e_rmp2, 2)

    def test_uoomp2(self):
        mp2 = mp.OOMP2(self.uhf).run()
        self.assertAlmostEqual(mp2.e_corr, self.e_ump2, 2)


if __name__ == '__main__':
    unittest.main()
