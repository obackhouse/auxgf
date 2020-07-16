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
        self.rhf_df = hf.RHF(self.m, with_df=True).run()
        self.uhf_df = hf.UHF(self.m, with_df=True).run()
        self.eri_rhf = self.rhf.eri_mo
        self.eri_uhf = self.uhf.eri_mo
        self.eri_rhf_df = self.rhf_df.eri_mo
        self.eri_uhf_df = self.uhf_df.eri_mo

    def test_build_rmp2_part(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri_rhf, self.rhf.chempot)
        e1, v1 = aux.mp2.build_mp2_part(eo, ev, xija, wtol=0)
        e2, v2 = aux.rmp2.build_rmp2_part(eo, ev, xija, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(e1 - e2)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(v1 - v2)), 0, 12)

    def test_build_ump2_part(self):
        eo, ev, xija, xabi = aux.ump2._parse_uhf(self.uhf.e, self.eri_uhf[0], self.uhf.chempot)
        e1, v1 = aux.mp2.build_mp2_part(eo, ev, xija, wtol=0)
        e2, v2 = aux.ump2.build_ump2_part(eo, ev, xija, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(e1 - e2)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(v1 - v2)), 0, 12)

    def test_build_rmp2(self):
        se1 = aux.mp2.build_mp2(self.rhf.e, self.eri_rhf, chempot=self.rhf.chempot, wtol=0)
        se2 = aux.rmp2.build_rmp2(self.rhf.e, self.eri_rhf, chempot=self.rhf.chempot, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(se1.e - se2.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(se1.v - se2.v)), 0, 12)

    def test_build_ump2(self):
        se1a = aux.mp2.build_mp2(self.uhf.e, self.eri_uhf[0], chempot=self.uhf.chempot, wtol=0)
        se2a = aux.ump2.build_ump2(self.uhf.e, self.eri_uhf[0], chempot=self.uhf.chempot, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(se1a.e - se2a.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(se1a.v - se2a.v)), 0, 12)
        se1a, se1b = aux.mp2.build_mp2(self.uhf.e, self.eri_uhf, chempot=self.uhf.chempot, wtol=0)
        se2a = aux.ump2.build_ump2(self.uhf.e, self.eri_uhf[0], chempot=self.uhf.chempot, wtol=0)
        se2b = aux.ump2.build_ump2(self.uhf.e[::-1], self.eri_uhf[1][::-1], chempot=self.uhf.chempot[::-1], wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(se1a.e - se2a.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(se1a.v - se2a.v)), 0, 12)

    def test_build_rmp2_iter(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri_rhf, self.rhf.chempot)
        se_null = aux.Aux([], [[],]*self.rhf.nao, chempot=self.rhf.chempot)
        se1 = aux.mp2.build_mp2_iter(se_null, self.rhf.fock_mo, self.eri_rhf, wtol=0)
        se2 = aux.rmp2.build_rmp2_iter(se_null, self.rhf.fock_mo, self.eri_rhf, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(se1.e - se2.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(se1.v, se1.v.T) - np.dot(se2.v, se2.v.T))), 0, 12)

    def test_build_ump2_iter(self):
        eo, ev, xija, xabi = aux.ump2._parse_uhf(self.uhf.e, self.eri_uhf[0], self.uhf.chempot)
        sea_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[0])
        seb_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[1])
        se1a, se1b = aux.mp2.build_mp2_iter((sea_null, seb_null), self.uhf.fock_mo, self.eri_uhf, wtol=0)
        se2a, se2b = aux.ump2.build_ump2_iter((sea_null, seb_null), self.uhf.fock_mo, self.eri_uhf, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(se1a.e - se2a.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(se1b.e - se2b.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(se1a.v, se1a.v.T) - np.dot(se2a.v, se2a.v.T))), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(se1b.v, se1b.v.T) - np.dot(se2b.v, se2b.v.T))), 0, 12)

    def test_build_dfrmp2_part(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_rhf_df, self.eri_rhf_df, self.rhf_df.chempot)
        e1, v1 = aux.mp2.build_dfmp2_part(eo, ev, ixq, qja, wtol=0)
        e2, v2 = aux.dfrmp2.build_dfrmp2_part(eo, ev, ixq, qja, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(e1 - e2)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(v1 - v2)), 0, 12)

    def test_build_dfump2_part(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_uhf_df, self.eri_uhf_df, self.uhf_df.chempot)
        e1, v1 = aux.mp2.build_dfmp2_part(eo, ev, ixq, qja, wtol=0)
        e2, v2 = aux.dfump2.build_dfump2_part(eo, ev, ixq, qja, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(e1 - e2)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(v1 - v2)), 0, 12)

    def test_build_dfrmp2(self):
        se1 = aux.mp2.build_dfmp2(self.rhf_df.e, self.eri_rhf_df, self.eri_rhf_df, chempot=self.rhf_df.chempot, wtol=0)
        se2 = aux.dfrmp2.build_dfrmp2(self.rhf_df.e, self.eri_rhf_df, self.eri_rhf_df, chempot=self.rhf_df.chempot, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(se1.e - se2.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(se1.v - se2.v)), 0, 12)

    def test_build_dfump2(self):
        se1a, se1b = aux.mp2.build_dfmp2(self.uhf_df.e, self.eri_uhf_df, self.eri_uhf_df, chempot=self.uhf_df.chempot, wtol=0)
        se2a = aux.dfump2.build_dfump2(self.uhf_df.e, self.eri_uhf_df, self.eri_uhf_df, chempot=self.uhf_df.chempot, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(se1a.e - se2a.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(se1a.v - se2a.v)), 0, 12)

    def test_build_dfrmp2_iter(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_rhf_df, self.eri_rhf_df, self.rhf_df.chempot)
        se_null = aux.Aux([], [[],]*self.rhf_df.nao, chempot=self.rhf_df.chempot)
        se1 = aux.mp2.build_dfmp2_iter(se_null, self.rhf_df.fock_mo, self.eri_rhf_df, wtol=0)
        se2 = aux.dfrmp2.build_dfrmp2_iter(se_null, self.rhf_df.fock_mo, self.eri_rhf_df, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(se1.e - se2.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(se1.v, se1.v.T) - np.dot(se2.v, se2.v.T))), 0, 12)

    def test_build_dfump2_iter(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_uhf_df, self.eri_uhf_df, self.uhf_df.chempot)
        sea_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[0])
        seb_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[1])
        se1a, se1b = aux.mp2.build_dfmp2_iter((sea_null, seb_null), self.uhf_df.fock_mo, self.eri_uhf_df, wtol=0)
        se2a, se2b = aux.dfump2.build_dfump2_iter((sea_null, seb_null), self.uhf_df.fock_mo, self.eri_uhf_df, wtol=0)
        self.assertAlmostEqual(np.max(np.absolute(se1a.e - se2a.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(se1b.e - se2b.e)), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(se1a.v, se1a.v.T) - np.dot(se2a.v, se2a.v.T))), 0, 12)
        self.assertAlmostEqual(np.max(np.absolute(np.dot(se1b.v, se1b.v.T) - np.dot(se2b.v, se2b.v.T))), 0, 12)


if __name__ == '__main__':
    unittest.main()
