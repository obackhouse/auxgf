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
        self.eri = self.rhf.eri_mo
        self.se = aux.build_rmp2(self.rhf.e, self.eri, chempot=self.rhf.chempot, wtol=0)
        self.e_mp2 = -0.20905684700662164
        self.e_mp2_scs = -0.20503556854447708

    def get_emp2(self, e, v, mask):
        fac = 1 if np.all(self.rhf.e[mask] < self.rhf.chempot) else -1
        return util.einsum('xk,xk->', v[mask]**2, fac / (self.rhf.e[mask,None] - e[None,:]))

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.eri, self.se, self.e_mp2, self.e_mp2_scs, self.get_emp2

    def test_parser(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        self.assertTrue(np.all(eo < self.rhf.chempot))
        self.assertTrue(np.all(ev >= self.rhf.chempot))
        self.assertTrue(xija.shape[1:] == (nocc, nocc, nvir))
        self.assertTrue(xabi.shape[1:] == (nvir, nvir, nocc))

    def test_make_coups(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        vv = np.outer(xija[:,1,2,3], 2*xija[:,1,2,3]-xija[:,2,1,3])
        v1 = aux.rmp2.make_coups_inner(vv)
        v2 = aux.rmp2.make_coups_outer(vv)
        self.assertAlmostEqual(np.max(np.absolute(v1 - v2)), 0, 8)

    def test_build_rmp2_part(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        e_occ, v_occ = aux.build_rmp2_part(eo, ev, xija, wtol=0)
        e_vir, v_vir = aux.build_rmp2_part(ev, eo, xabi, wtol=0)
        e_mp2_a = self.get_emp2(e_vir, v_vir, o)
        e_mp2_b = self.get_emp2(e_occ, v_occ, v)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 8)

    def test_build_rmp2(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        se_a = aux.build_rmp2(self.rhf.e, self.eri, chempot=self.rhf.chempot, wtol=0)
        e_mp2_a = self.get_emp2(se_a.e_vir, se_a.v_vir, o)
        e_mp2_b = self.get_emp2(se_a.e_occ, se_a.v_occ, v)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 8)

    def test_build_rmp2_iter(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        se_null = aux.Aux([], [[],]*self.rhf.nao, chempot=self.rhf.chempot)
        se_a = aux.build_rmp2_iter(se_null, self.rhf.fock_mo, self.eri, wtol=0)
        e_mp2_a = self.get_emp2(se_a.e_vir, se_a.v_vir, o)
        e_mp2_b = self.get_emp2(se_a.e_occ, se_a.v_occ, v)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 7)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 7)

    def test_build_rmp2_part_direct(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        e_occ, v_occ = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_rmp2_part_direct(eo, ev, xija, wtol=0)]))]
        e_vir, v_vir = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_rmp2_part_direct(ev, eo, xabi, wtol=0)]))]
        e_mp2_a = self.get_emp2(e_vir, v_vir, o)
        e_mp2_b = self.get_emp2(e_occ, v_occ, v)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 8)

    def test_build_rmp2_direct(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        se_a = [x for x in aux.build_rmp2_direct(self.rhf.e, self.eri, chempot=self.rhf.chempot, wtol=0)]
        se_a = sum(se_a, aux.Aux([], [[],]*self.rhf.nao))
        e_mp2_a = self.get_emp2(se_a.e_vir, se_a.v_vir, o)
        e_mp2_b = self.get_emp2(se_a.e_occ, se_a.v_occ, v)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 8)

    def test_build_rmp2_part_se_direct(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        imfq = grids.ImFqGrid(2**5, beta=2**3)
        se_a  = aux.build_rmp2_part_se_direct(eo, ev, xija, imfq, chempot=self.rhf.chempot)
        se_a += aux.build_rmp2_part_se_direct(ev, eo, xabi, imfq, chempot=self.rhf.chempot)
        self.assertAlmostEqual(np.max(np.absolute(se_a - self.se.as_spectrum(imfq))), 0, 8)

    def test_build_rmp2_se_direct(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        imfq = grids.ImFqGrid(2**5, beta=2**3)
        se_a = aux.build_rmp2_se_direct(self.rhf.e, self.eri, imfq, chempot=self.rhf.chempot)
        self.assertAlmostEqual(np.max(np.absolute(se_a - self.se.as_spectrum(imfq))), 0, 8)

    def test_scs_build_rmp2(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        se_a = aux.build_rmp2(self.rhf.e, self.eri, chempot=self.rhf.chempot, wtol=0, os_factor=1.2, ss_factor=0.33)
        e_mp2_a = self.get_emp2(se_a.e_vir, se_a.v_vir, o)
        e_mp2_b = self.get_emp2(se_a.e_occ, se_a.v_occ, v)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_b, 8)

    def test_scs_build_rmp2_iter(self):
        eo, ev, xija, xabi = aux.rmp2._parse_rhf(self.rhf.e, self.eri, self.rhf.chempot)
        nocc, nvir = eo.size, ev.size
        o, v = slice(None, nocc), slice(nocc, None)
        se_null = aux.Aux([], [[],]*self.rhf.nao, chempot=self.rhf.chempot)
        se_a = aux.build_rmp2_iter(se_null, self.rhf.fock_mo, self.eri, wtol=0, os_factor=1.2, ss_factor=0.33)
        e_mp2_a = self.get_emp2(se_a.e_vir, se_a.v_vir, o)
        e_mp2_b = self.get_emp2(se_a.e_occ, se_a.v_occ, v)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_a, 7)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_b, 7)


if __name__ == '__main__':
    unittest.main()
