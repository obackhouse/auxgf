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
        self.rhf_df = hf.RHF(self.m, with_df=True).run()
        self.eri = self.rhf.eri_mo
        self.eri_df = self.rhf_df.eri_mo
        self.se = aux.build_rmp2(self.rhf_df.e, util.einsum('qij,qkl->ijkl', self.eri_df, self.eri_df), chempot=self.rhf_df.chempot, wtol=0)
        self.e_mp2 = -0.20905684700662164
        self.e_mp2_scs = -0.20503556854447708

    def get_emp2(self, e, v, mask):
        fac = 1 if np.all(self.rhf.e[mask] < self.rhf.chempot) else -1
        return util.einsum('xk,xk->', v[mask]**2, fac / (self.rhf.e[mask,None] - e[None,:])).ravel()[0]

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.rhf_df, self.eri, self.eri_df, self.se, self.e_mp2, self.e_mp2_scs

    def test_parser(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot)

    def test_build_dfrmp2_part(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        e_occ, v_occ = aux.build_dfrmp2_part(eo, ev, ixq, qja, wtol=0)
        e_vir, v_vir = aux.build_dfrmp2_part(ev, eo, axq, qbi, wtol=0)
        e_mp2_a = self.get_emp2(e_vir, v_vir, o)
        e_mp2_b = self.get_emp2(e_occ, v_occ, v)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 4)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 4)

    def test_build_dfrmp2(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        e_occ, v_occ = aux.build_dfrmp2_part(eo, ev, ixq, qja, wtol=0)
        e_vir, v_vir = aux.build_dfrmp2_part(ev, eo, axq, qbi, wtol=0)
        se = aux.build_dfrmp2(self.rhf_df.e, self.eri_df, self.eri_df, chempot=self.rhf.chempot, wtol=0)
        e_mp2_a = self.get_emp2(se.e_vir, se.v_vir, o)
        e_mp2_b = self.get_emp2(se.e_occ, se.v_occ, v)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 4)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 4)

    def test_build_dfrmp2_iter(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        se_null = aux.Aux([], [[],]*self.rhf.nao, chempot=self.rhf_df.chempot)
        se = aux.build_dfrmp2_iter(se_null, self.rhf_df.fock_mo, self.eri_df, wtol=0)
        e_mp2_a = self.get_emp2(se.e_vir, se.v_vir, o)
        e_mp2_b = self.get_emp2(se.e_occ, se.v_occ, v)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 4)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 4)

    def test_build_dfrmp2_part_direct(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        e_occ, v_occ = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_dfrmp2_part_direct(eo, ev, ixq, qja, wtol=0)]))]
        e_vir, v_vir = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_dfrmp2_part_direct(ev, eo, axq, qbi, wtol=0)]))]
        e_mp2_a = self.get_emp2(e_vir, v_vir, o)
        e_mp2_b = self.get_emp2(e_occ, v_occ, v)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 4)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 4)

    def test_build_dfrmp2_direct(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        se = [x for x in aux.build_dfrmp2_direct(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot, wtol=0)]
        se = sum(se, aux.Aux([], [[],]*self.rhf.nao))
        e_mp2_a = self.get_emp2(se.e_vir, se.v_vir, o)
        e_mp2_b = self.get_emp2(se.e_occ, se.v_occ, v)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 4)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 4)

    def test_build_dfrmp2_part_se_direct(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        imfq = grids.ImFqGrid(2**5, beta=2**3)
        se  = aux.build_dfrmp2_part_se_direct(eo, ev, ixq, qja, imfq, chempot=self.rhf_df.chempot)
        se += aux.build_dfrmp2_part_se_direct(ev, eo, axq, qbi, imfq, chempot=self.rhf_df.chempot)
        self.assertAlmostEqual(np.max(np.absolute(se - self.se.as_spectrum(imfq))), 0, 4)

    def test_build_dfrmp2_se_direct(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        imfq = grids.ImFqGrid(2**5, beta=2**3)
        se = aux.build_dfrmp2_se_direct(self.rhf_df.e, self.eri_df, self.eri_df, imfq, chempot=self.rhf_df.chempot)
        self.assertAlmostEqual(np.max(np.absolute(se - self.se.as_spectrum(imfq))), 0, 4)

    def test_scs_build_dfrmp2(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        se = aux.build_dfrmp2(self.rhf_df.e, self.eri_df, self.eri_df, chempot=self.rhf_df.chempot, wtol=0, os_factor=1.2, ss_factor=0.33)
        e_mp2_a = self.get_emp2(se.e_vir, se.v_vir, o)
        e_mp2_b = self.get_emp2(se.e_occ, se.v_occ, v)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_a, 3)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_b, 3)

    def test_scs_build_dfrmp2_iter(self):
        eo, ev, ixq, qja, axq, qbi = aux.dfrmp2._parse_rhf(self.rhf_df.e, self.eri_df, self.eri_df, self.rhf_df.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        se_null = aux.Aux([], [[],]*self.rhf.nao, chempot=self.rhf_df.chempot)
        se = aux.build_dfrmp2_iter(se_null, self.rhf_df.fock_mo, self.eri_df, wtol=0, os_factor=1.2, ss_factor=0.33)
        e_mp2_a = self.get_emp2(se.e_vir, se.v_vir, o)
        e_mp2_b = self.get_emp2(se.e_occ, se.v_occ, v)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_a, 3)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_b, 3)


if __name__ == '__main__':
    unittest.main()
