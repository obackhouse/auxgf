import unittest
import numpy as np

from auxgf import mol, hf, dft


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')

    @classmethod
    def tearDownClass(self):
        del self.m

    def test_rdft(self):
        rdft = dft.RDFT(self.m).run()
        self.assertAlmostEqual(rdft.e_tot, -75.85119020953266, 8)

    def test_udft(self):
        udft = dft.UDFT(self.m).run()
        self.assertAlmostEqual(udft.e_tot, -75.85119020952622, 8)


if __name__ == '__main__':
    unittest.main()
