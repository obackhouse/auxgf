import unittest
import numpy as np

from auxgf import util, mol, hf, aux, grids


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')
        self.rhf = hf.RHF(self.m).run()
        self.uhf = hf.UHF(self.m).run()
        self.gf = aux.Aux(self.rhf.e, np.eye(self.rhf.nao), chempot=self.rhf.chempot)
        self.se = aux.build_rmp2(self.rhf.e, self.rhf.eri_mo, chempot=self.rhf.chempot)
        self.gf_a = aux.Aux(self.uhf.e[0], np.eye(self.uhf.nao), chempot=self.uhf.chempot[0])
        self.gf_b = aux.Aux(self.uhf.e[1], np.eye(self.uhf.nao), chempot=self.uhf.chempot[1])
        self.se_a = aux.build_ump2(self.uhf.e, self.uhf.eri_mo[0], chempot=self.uhf.chempot)
        self.se_b = aux.build_ump2(self.uhf.e[::-1], self.uhf.eri_mo[1][::-1], chempot=self.uhf.chempot[::-1])
        self.e_rmp2 = -0.20905684700662164
        self.e_ump2 = -0.20905685057662993

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.uhf, self.gf, self.se, self.gf_a, self.gf_b, self.se_a, self.se_b, self.e_rmp2, self.e_ump2

    def test_rmp2_energy(self):
        e_mp2_a = aux.energy_mp2_aux(self.rhf.e, self.se)
        e_mp2_b = aux.energy_mp2_aux(self.rhf.e, self.se, both_sides=True)
        self.assertAlmostEqual(self.e_rmp2, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_rmp2, e_mp2_b, 8)

    def test_ump2_energy(self):
        e_mp2_a  = aux.energy_mp2_aux(self.uhf.e[0], self.se_a) * 0.5
        e_mp2_a += aux.energy_mp2_aux(self.uhf.e[1], self.se_b) * 0.5 
        e_mp2_b  = aux.energy_mp2_aux(self.uhf.e[0], self.se_a, both_sides=True) * 0.5
        e_mp2_b += aux.energy_mp2_aux(self.uhf.e[1], self.se_b, both_sides=True) * 0.5 
        self.assertAlmostEqual(self.e_ump2, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_ump2, e_mp2_b, 8)

    def test_r2b_energy(self):
        e_mp2_a = aux.energy_2body_aux(self.gf, self.se)
        e_mp2_b = aux.energy_2body_aux(self.gf, self.se, both_sides=True)
        self.assertAlmostEqual(self.e_rmp2, 0.5 * e_mp2_a, 8)
        self.assertAlmostEqual(self.e_rmp2, 0.5 * e_mp2_b, 8)

    def test_u2b_energy(self):
        e_mp2_a  = aux.energy_2body_aux(self.gf_a, self.se_a) * 0.5
        e_mp2_a += aux.energy_2body_aux(self.gf_b, self.se_b) * 0.5
        e_mp2_b  = aux.energy_2body_aux(self.gf_a, self.se_a, both_sides=True) * 0.5
        e_mp2_b += aux.energy_2body_aux(self.gf_b, self.se_b, both_sides=True) * 0.5
        self.assertAlmostEqual(self.e_rmp2, 0.5 * e_mp2_a, 8)
        self.assertAlmostEqual(self.e_rmp2, 0.5 * e_mp2_b, 8)


if __name__ == '__main__':
    unittest.main()
