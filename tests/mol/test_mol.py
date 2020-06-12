import unittest
import numpy as np

from auxgf import mol


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')

    @classmethod
    def tearDownClass(self):
        del self.m

    def test_intor(self):
        s1 = self.m._pyscf.get_ovlp()
        s2 = self.m.intor('int1e_ovlp')
        self.assertAlmostEqual(np.max(np.absolute(s1 - s2)), 0, 8)

    def test_ncore(self):
        self.assertEqual(self.m.ncore, 2)

    def test_properties(self):
        self.assertEqual(self.m.natom, 3)
        self.assertEqual(self.m.nelec, 10)
        self.assertEqual(self.m.nalph, 5)
        self.assertEqual(self.m.nbeta, 5)
        self.assertEqual(tuple(self.m.labels), ('O', 'H', 'H'))
        self.assertAlmostEqual(np.max(np.absolute(np.array(self.m.coords) - np.array([self.m._pyscf.atom_coord(i) for i in range(self.m.natom)]))), 0, 8)
        self.assertEqual(self.m.atoms, 'O 0 0 0; H 0 0 1; H 0 1 0')
        self.assertEqual(self.m.charge, 0)
        self.assertEqual(tuple(self.m.charges), (8, 1, 1))
        self.assertEqual(self.m.spin, 0)


if __name__ == '__main__':
    unittest.main()
