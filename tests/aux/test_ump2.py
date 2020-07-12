import unittest
import numpy as np

from auxgf import util, mol, hf, aux, grids


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1', basis='cc-pvdz', spin=1)
        self.uhf = hf.UHF(self.m).run()
        self.eri = self.uhf.eri_mo
        self.se = (aux.build_ump2(self.uhf.e, self.eri[0], chempot=self.uhf.chempot, wtol=0),
                   aux.build_ump2(self.uhf.e[::-1], self.eri[1][::-1], chempot=self.uhf.chempot[::-1], wtol=0))
        self.e_mp2 = -0.15197757655845123
        self.e_mp2_scs = -0.1502359064727459

    def get_emp2(self, e, v, mask, spin):
        fac = 0.5 if np.all(self.uhf.e[spin][mask] < self.uhf.chempot[spin]) else -0.5
        return util.einsum('xk,xk->', v[mask]**2, fac / (self.uhf.e[spin][mask,None] - e[None,:]))

    @classmethod
    def tearDownClass(self):
        del self.m, self.uhf, self.eri, self.se, self.e_mp2, self.e_mp2_scs

    def test_parser(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        self.assertTrue(np.all(eoa < self.uhf.chempot[0]))
        self.assertTrue(np.all(eob < self.uhf.chempot[1]))
        self.assertTrue(np.all(eva >= self.uhf.chempot[0]))
        self.assertTrue(np.all(evb >= self.uhf.chempot[1]))
        self.assertTrue(xija_aa.shape[1:] == (nocca, nocca, nvira))
        self.assertTrue(xija_ab.shape[1:] == (nocca, noccb, nvirb))
        self.assertTrue(xija_ba.shape[1:] == (noccb, nocca, nvira))
        self.assertTrue(xija_bb.shape[1:] == (noccb, noccb, nvirb))
        self.assertTrue(xabi_aa.shape[1:] == (nvira, nvira, nocca))
        self.assertTrue(xabi_ab.shape[1:] == (nvira, nvirb, noccb))
        self.assertTrue(xabi_ba.shape[1:] == (nvirb, nvira, nocca))
        self.assertTrue(xabi_bb.shape[1:] == (nvirb, nvirb, noccb))

    def test_make_coups(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        vv = np.outer(xija_aa[:,1,2,3], 2*xija_aa[:,1,2,3]-xija_aa[:,2,1,3])
        v1 = aux.ump2.make_coups_inner(vv)
        v2 = aux.ump2.make_coups_outer(vv)
        self.assertAlmostEqual(np.max(np.absolute(v1 - v2)), 0, 8)

    def test_build_ump2_part(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        e_occ_a, v_occ_a = aux.build_ump2_part((eoa, eob), (eva, evb), (xija_aa, xija_ab), wtol=0)
        e_occ_b, v_occ_b = aux.build_ump2_part((eob, eoa), (evb, eva), (xija_bb, xija_ba), wtol=0)
        e_vir_a, v_vir_a = aux.build_ump2_part((eva, evb), (eoa, eob), (xabi_aa, xabi_ab), wtol=0)
        e_vir_b, v_vir_b = aux.build_ump2_part((evb, eva), (eob, eoa), (xabi_bb, xabi_ba), wtol=0)
        e_mp2_a  = self.get_emp2(e_vir_a, v_vir_a, oa, 0)
        e_mp2_a += self.get_emp2(e_vir_b, v_vir_b, ob, 1)
        e_mp2_b  = self.get_emp2(e_occ_a, v_occ_a, va, 0)
        e_mp2_b += self.get_emp2(e_occ_b, v_occ_b, vb, 1)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 8)

    def test_build_ump2(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        se_a = aux.build_ump2(self.uhf.e, self.eri[0], chempot=self.uhf.chempot, wtol=0)
        se_b = aux.build_ump2(self.uhf.e[::-1], self.eri[1][::-1], chempot=self.uhf.chempot[::-1], wtol=0)
        e_mp2_a  = self.get_emp2(se_a.e_vir, se_a.v_vir, oa, 0)
        e_mp2_a += self.get_emp2(se_b.e_vir, se_b.v_vir, ob, 1)
        e_mp2_b  = self.get_emp2(se_a.e_occ, se_a.v_occ, va, 0)
        e_mp2_b += self.get_emp2(se_b.e_occ, se_b.v_occ, vb, 1)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 8)

    def test_build_ump2_iter(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        se_a_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[0])
        se_b_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[1])
        se_a, se_b = aux.build_ump2_iter((se_a_null, se_b_null), self.uhf.fock_mo, self.eri, wtol=0)
        e_mp2_a  = self.get_emp2(se_a.e_vir, se_a.v_vir, oa, 0)
        e_mp2_a += self.get_emp2(se_b.e_vir, se_b.v_vir, ob, 1)
        e_mp2_b  = self.get_emp2(se_a.e_occ, se_a.v_occ, va, 0)
        e_mp2_b += self.get_emp2(se_b.e_occ, se_b.v_occ, vb, 1)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 7)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 7)

    def test_build_ump2_part_direct(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        e_occ_a, v_occ_a = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_ump2_part_direct((eoa, eob), (eva, evb), (xija_aa, xija_ab), wtol=0)]))]
        e_occ_b, v_occ_b = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_ump2_part_direct((eob, eoa), (evb, eva), (xija_bb, xija_ba), wtol=0)]))]
        e_vir_a, v_vir_a = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_ump2_part_direct((eva, evb), (eoa, eob), (xabi_aa, xabi_ab), wtol=0)]))]
        e_vir_b, v_vir_b = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_ump2_part_direct((evb, eva), (eob, eoa), (xabi_bb, xabi_ba), wtol=0)]))]
        e_mp2_a  = self.get_emp2(e_vir_a, v_vir_a, oa, 0)
        e_mp2_a += self.get_emp2(e_vir_b, v_vir_b, ob, 1)
        e_mp2_b  = self.get_emp2(e_occ_a, v_occ_a, va, 0)
        e_mp2_b += self.get_emp2(e_occ_b, v_occ_b, vb, 1)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 8)

    def test_build_ump2_direct(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        se_a_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[0])
        se_b_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[1])
        se_a = [x for x in aux.build_ump2_direct(self.uhf.e, self.eri[0], chempot=self.uhf.chempot, wtol=0)]
        se_b = [x for x in aux.build_ump2_direct(self.uhf.e[::-1], self.eri[1][::-1], chempot=self.uhf.chempot[::-1], wtol=0)]
        se_a = sum(se_a, se_a_null)
        se_b = sum(se_b, se_b_null)
        e_mp2_a  = self.get_emp2(se_a.e_vir, se_a.v_vir, oa, 0)
        e_mp2_a += self.get_emp2(se_b.e_vir, se_b.v_vir, ob, 1)
        e_mp2_b  = self.get_emp2(se_a.e_occ, se_a.v_occ, va, 0)
        e_mp2_b += self.get_emp2(se_b.e_occ, se_b.v_occ, vb, 1)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 8)

    def test_build_ump2_part_se_direct(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        imfq = grids.ImFqGrid(2**5, beta=2**3)
        se_a  = aux.build_ump2_part_se_direct((eoa, eob), (eva, evb), (xija_aa, xija_ab), imfq, chempot=self.uhf.chempot)
        se_a += aux.build_ump2_part_se_direct((eva, evb), (eoa, eob), (xabi_aa, xabi_ab), imfq, chempot=self.uhf.chempot)
        se_b  = aux.build_ump2_part_se_direct((eob, eoa), (evb, eva), (xija_bb, xija_ba), imfq, chempot=self.uhf.chempot[::-1])
        se_b += aux.build_ump2_part_se_direct((evb, eva), (eob, eoa), (xabi_bb, xabi_ba), imfq, chempot=self.uhf.chempot[::-1])
        self.assertAlmostEqual(np.max(np.absolute(se_a - self.se[0].as_spectrum(imfq))), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(se_b - self.se[1].as_spectrum(imfq))), 0, 8)

    def test_build_ump2_se_direct(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        imfq = grids.ImFqGrid(2**5, beta=2**3)
        se_a = aux.build_ump2_se_direct(self.uhf.e, self.eri[0], imfq, chempot=self.uhf.chempot)
        se_b = aux.build_ump2_se_direct(self.uhf.e[::-1], self.eri[1][::-1], imfq, chempot=self.uhf.chempot[::-1])
        self.assertAlmostEqual(np.max(np.absolute(se_a - self.se[0].as_spectrum(imfq))), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(se_b - self.se[1].as_spectrum(imfq))), 0, 8)

    def test_scs_build_ump2(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        se_a = aux.build_ump2(self.uhf.e, self.eri[0], chempot=self.uhf.chempot, wtol=0, os_factor=1.2, ss_factor=0.33)
        se_b = aux.build_ump2(self.uhf.e[::-1], self.eri[1][::-1], chempot=self.uhf.chempot[::-1], wtol=0, os_factor=1.2, ss_factor=0.33)
        e_mp2_a  = self.get_emp2(se_a.e_vir, se_a.v_vir, oa, 0)
        e_mp2_a += self.get_emp2(se_b.e_vir, se_b.v_vir, ob, 1)
        e_mp2_b  = self.get_emp2(se_a.e_occ, se_a.v_occ, va, 0)
        e_mp2_b += self.get_emp2(se_b.e_occ, se_b.v_occ, vb, 1)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_a, 8)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_b, 8)

    def test_scs_build_ump2_iter(self):
        (eoa, eob), (eva, evb), (xija_aa, xija_ab), (xabi_aa, xabi_ab) = aux.ump2._parse_uhf(self.uhf.e, self.eri[0], self.uhf.chempot)
        _, _, (xija_bb, xija_ba), (xabi_bb, xabi_ba) = aux.ump2._parse_uhf(self.uhf.e[::-1], self.eri[1][::-1], self.uhf.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        se_a_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[0])
        se_b_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[1])
        se_a, se_b = aux.build_ump2_iter((se_a_null, se_b_null), self.uhf.fock_mo, self.eri, wtol=0, os_factor=1.2, ss_factor=0.33)
        e_mp2_a  = self.get_emp2(se_a.e_vir, se_a.v_vir, oa, 0)
        e_mp2_a += self.get_emp2(se_b.e_vir, se_b.v_vir, ob, 1)
        e_mp2_b  = self.get_emp2(se_a.e_occ, se_a.v_occ, va, 0)
        e_mp2_b += self.get_emp2(se_b.e_occ, se_b.v_occ, vb, 1)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_a, 7)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_b, 7)


if __name__ == '__main__':
    unittest.main()
