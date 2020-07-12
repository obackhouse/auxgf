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
        self.uhf_df = hf.UHF(self.m, with_df=True).run()
        self.eri = self.uhf.eri_mo
        self.eri_df = self.uhf_df.eri_mo
        self.se = (aux.build_ump2(self.uhf_df.e, util.einsum('qij,sqkl->sijkl', self.eri_df[0], self.eri_df), chempot=self.uhf_df.chempot, wtol=0),
                   aux.build_ump2(self.uhf_df.e[::-1], util.einsum('qij,sqkl->sijkl', self.eri_df[1], self.eri_df[::-1]), chempot=self.uhf.chempot[::-1], wtol=0))
        self.e_mp2 = -0.15197757655845123
        self.e_mp2_scs = -0.1502359064727459

    def get_emp2(self, e, v, mask, spin):
        fac = 0.5 if np.all(self.uhf.e[spin][mask] < self.uhf.chempot[spin]) else -0.5
        return util.einsum('xk,xk->', v[mask]**2, fac / (self.uhf.e[spin][mask,None] - e[None,:])).ravel()[0]

    @classmethod
    def tearDownClass(self):
        del self.m, self.uhf, self.uhf_df, self.eri, self.eri_df, self.se, self.e_mp2, self.e_mp2_scs

    def test_parser(self):
        (eoa, eob), (eva, evb), (ixq_a, _), (qja_a, qja_b), (axq_a, _), (qbi_a, qbi_b) = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_df, self.eri_df, self.uhf_df.chempot)
        (eob, eoa), (evb, eva), (ixq_b, _), (qja_b, qja_a), (axq_b, _), (qbi_b, qbi_a) = aux.dfump2._parse_uhf(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], self.uhf_df.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        self.assertTrue(np.all(eoa < self.uhf_df.chempot[0]))
        self.assertTrue(np.all(eob < self.uhf_df.chempot[1]))
        self.assertTrue(np.all(eva >= self.uhf_df.chempot[0]))
        self.assertTrue(np.all(evb >= self.uhf_df.chempot[1]))
        self.assertTrue(ixq_a.shape[0] == nocca)
        self.assertTrue(ixq_b.shape[0] == noccb)
        self.assertTrue(axq_a.shape[0] == nvira)
        self.assertTrue(axq_b.shape[0] == nvirb)
        self.assertTrue(qja_a.shape[1:] == (nocca, nvira))
        self.assertTrue(qja_b.shape[1:] == (noccb, nvirb))
        self.assertTrue(qbi_a.shape[1:] == (nvira, nocca))
        self.assertTrue(qbi_b.shape[1:] == (nvirb, noccb))

    def test_build_dfump2_part(self):
        (eoa, eob), (eva, evb), (ixq_a, _), (qja_a, qja_b), (axq_a, _), (qbi_a, qbi_b) = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_df, self.eri_df, self.uhf_df.chempot)
        (eob, eoa), (evb, eva), (ixq_b, _), (qja_b, qja_a), (axq_b, _), (qbi_b, qbi_a) = aux.dfump2._parse_uhf(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], self.uhf_df.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        e_occ_a, v_occ_a = aux.build_dfump2_part((eoa, eob), (eva, evb), (ixq_a,), (qja_a, qja_b), wtol=0)
        e_occ_b, v_occ_b = aux.build_dfump2_part((eob, eoa), (evb, eva), (ixq_b,), (qja_b, qja_a), wtol=0)
        e_vir_a, v_vir_a = aux.build_dfump2_part((eva, evb), (eoa, eob), (axq_a,), (qbi_a, qbi_b), wtol=0)
        e_vir_b, v_vir_b = aux.build_dfump2_part((evb, eva), (eob, eoa), (axq_b,), (qbi_b, qbi_a), wtol=0)
        e_mp2_a  = self.get_emp2(e_vir_a, v_vir_a, oa, 0)
        e_mp2_a += self.get_emp2(e_vir_b, v_vir_b, ob, 1)
        e_mp2_b  = self.get_emp2(e_occ_a, v_occ_a, va, 0)
        e_mp2_b += self.get_emp2(e_occ_b, v_occ_b, vb, 1)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 4)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 4)

    def test_build_dfump2(self):
        (eoa, eob), (eva, evb), (ixq_a, _), (qja_a, qja_b), (axq_a, _), (qbi_a, qbi_b) = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_df, self.eri_df, self.uhf_df.chempot)
        (eob, eoa), (evb, eva), (ixq_b, _), (qja_b, qja_a), (axq_b, _), (qbi_b, qbi_a) = aux.dfump2._parse_uhf(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], self.uhf_df.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        se_a = aux.build_dfump2(self.uhf_df.e, self.eri_df, self.eri_df, chempot=self.uhf.chempot, wtol=0)
        se_b = aux.build_dfump2(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], chempot=self.uhf.chempot[::-1], wtol=0)
        e_mp2_a  = self.get_emp2(se_a.e_vir, se_a.v_vir, oa, 0)
        e_mp2_a += self.get_emp2(se_b.e_vir, se_b.v_vir, ob, 1)
        e_mp2_b  = self.get_emp2(se_a.e_occ, se_a.v_occ, va, 0)
        e_mp2_b += self.get_emp2(se_b.e_occ, se_b.v_occ, vb, 1)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 4)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 4)

    def test_build_dfump2_iter(self):
        (eoa, eob), (eva, evb), (ixq_a, _), (qja_a, qja_b), (axq_a, _), (qbi_a, qbi_b) = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_df, self.eri_df, self.uhf_df.chempot)
        (eob, eoa), (evb, eva), (ixq_b, _), (qja_b, qja_a), (axq_b, _), (qbi_b, qbi_a) = aux.dfump2._parse_uhf(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], self.uhf_df.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        se_a_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[0])
        se_b_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[1])
        se_a, se_b = aux.build_dfump2_iter((se_a_null, se_b_null), self.uhf.fock_mo, self.eri_df, wtol=0)
        e_mp2_a  = self.get_emp2(se_a.e_vir, se_a.v_vir, oa, 0)
        e_mp2_a += self.get_emp2(se_b.e_vir, se_b.v_vir, ob, 1)
        e_mp2_b  = self.get_emp2(se_a.e_occ, se_a.v_occ, va, 0)
        e_mp2_b += self.get_emp2(se_b.e_occ, se_b.v_occ, vb, 1)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 3)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 3)

    def test_build_dfump2_part_direct(self):
        (eoa, eob), (eva, evb), (ixq_a, _), (qja_a, qja_b), (axq_a, _), (qbi_a, qbi_b) = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_df, self.eri_df, self.uhf_df.chempot)
        (eob, eoa), (evb, eva), (ixq_b, _), (qja_b, qja_a), (axq_b, _), (qbi_b, qbi_a) = aux.dfump2._parse_uhf(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], self.uhf_df.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        e_occ_a, v_occ_a = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_dfump2_part_direct((eoa, eob), (eva, evb), (ixq_a,), (qja_a, qja_b), wtol=0)]))]
        e_occ_b, v_occ_b = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_dfump2_part_direct((eob, eoa), (evb, eva), (ixq_b,), (qja_b, qja_a), wtol=0)]))]
        e_vir_a, v_vir_a = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_dfump2_part_direct((eva, evb), (eoa, eob), (axq_a,), (qbi_a, qbi_b), wtol=0)]))]
        e_vir_b, v_vir_b = [np.block(list(x)) for x in list(zip(*[(e,v) for e,v in aux.build_dfump2_part_direct((evb, eva), (eob, eoa), (axq_b,), (qbi_b, qbi_a), wtol=0)]))]
        e_mp2_a  = self.get_emp2(e_vir_a, v_vir_a, oa, 0)
        e_mp2_a += self.get_emp2(e_vir_b, v_vir_b, ob, 1)
        e_mp2_b  = self.get_emp2(e_occ_a, v_occ_a, va, 0)
        e_mp2_b += self.get_emp2(e_occ_b, v_occ_b, vb, 1)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 4)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 4)

    def test_build_dfump2_direct(self):
        (eoa, eob), (eva, evb), (ixq_a, _), (qja_a, qja_b), (axq_a, _), (qbi_a, qbi_b) = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_df, self.eri_df, self.uhf_df.chempot)
        (eob, eoa), (evb, eva), (ixq_b, _), (qja_b, qja_a), (axq_b, _), (qbi_b, qbi_a) = aux.dfump2._parse_uhf(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], self.uhf_df.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        se_a_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf_df.chempot[0])
        se_b_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf_df.chempot[1])
        se_a = [x for x in aux.build_dfump2_direct(self.uhf_df.e, self.eri_df, self.eri_df, chempot=self.uhf_df.chempot, wtol=0)]
        se_b = [x for x in aux.build_dfump2_direct(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], chempot=self.uhf_df.chempot[::-1], wtol=0)]
        se_a = sum(se_a, se_a_null)
        se_b = sum(se_b, se_b_null)
        e_mp2_a  = self.get_emp2(se_a.e_vir, se_a.v_vir, oa, 0)
        e_mp2_a += self.get_emp2(se_b.e_vir, se_b.v_vir, ob, 1)
        e_mp2_b  = self.get_emp2(se_a.e_occ, se_a.v_occ, va, 0)
        e_mp2_b += self.get_emp2(se_b.e_occ, se_b.v_occ, vb, 1)
        self.assertAlmostEqual(self.e_mp2, e_mp2_a, 4)
        self.assertAlmostEqual(self.e_mp2, e_mp2_b, 4)

    def test_build_dfump2_part_se_direct(self):
        (eoa, eob), (eva, evb), (ixq_a, _), (qja_a, qja_b), (axq_a, _), (qbi_a, qbi_b) = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_df, self.eri_df, self.uhf_df.chempot)
        (eob, eoa), (evb, eva), (ixq_b, _), (qja_b, qja_a), (axq_b, _), (qbi_b, qbi_a) = aux.dfump2._parse_uhf(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], self.uhf_df.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        imfq = grids.ImFqGrid(2**5, beta=2**3)
        se_a  = aux.build_dfump2_part_se_direct((eoa, eob), (eva, evb), (ixq_a,), (qja_a, qja_b), imfq, chempot=self.uhf.chempot)
        se_a += aux.build_dfump2_part_se_direct((eva, evb), (eoa, eob), (axq_a,), (qbi_a, qbi_b), imfq, chempot=self.uhf.chempot)
        se_b  = aux.build_dfump2_part_se_direct((eob, eoa), (evb, eva), (ixq_b,), (qja_b, qja_a), imfq, chempot=self.uhf.chempot[::-1])
        se_b += aux.build_dfump2_part_se_direct((evb, eva), (eob, eoa), (axq_b,), (qbi_b, qbi_a), imfq, chempot=self.uhf.chempot[::-1])
        self.assertAlmostEqual(np.max(np.absolute(se_a - self.se[0].as_spectrum(imfq))), 0, 4)
        self.assertAlmostEqual(np.max(np.absolute(se_b - self.se[1].as_spectrum(imfq))), 0, 4)

    def test_build_dfump2_se_direct(self):
        (eoa, eob), (eva, evb), (ixq_a, _), (qja_a, qja_b), (axq_a, _), (qbi_a, qbi_b) = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_df, self.eri_df, self.uhf_df.chempot)
        (eob, eoa), (evb, eva), (ixq_b, _), (qja_b, qja_a), (axq_b, _), (qbi_b, qbi_a) = aux.dfump2._parse_uhf(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], self.uhf_df.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        imfq = grids.ImFqGrid(2**5, beta=2**3)
        se_a = aux.build_dfump2_se_direct(self.uhf_df.e, self.eri_df, self.eri_df, imfq, chempot=self.uhf.chempot)
        se_b = aux.build_dfump2_se_direct(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], imfq, chempot=self.uhf.chempot[::-1])
        self.assertAlmostEqual(np.max(np.absolute(se_a - self.se[0].as_spectrum(imfq))), 0, 4)
        self.assertAlmostEqual(np.max(np.absolute(se_b - self.se[1].as_spectrum(imfq))), 0, 4)

    def test_scs_build_dfump2(self):
        (eoa, eob), (eva, evb), (ixq_a, _), (qja_a, qja_b), (axq_a, _), (qbi_a, qbi_b) = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_df, self.eri_df, self.uhf_df.chempot)
        (eob, eoa), (evb, eva), (ixq_b, _), (qja_b, qja_a), (axq_b, _), (qbi_b, qbi_a) = aux.dfump2._parse_uhf(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], self.uhf_df.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        se_a = aux.build_dfump2(self.uhf_df.e, self.eri_df, self.eri_df, chempot=self.uhf.chempot, wtol=0, os_factor=1.2, ss_factor=0.33)
        se_b = aux.build_dfump2(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], chempot=self.uhf.chempot[::-1], wtol=0, os_factor=1.2, ss_factor=0.33)
        e_mp2_a  = self.get_emp2(se_a.e_vir, se_a.v_vir, oa, 0)
        e_mp2_a += self.get_emp2(se_b.e_vir, se_b.v_vir, ob, 1)
        e_mp2_b  = self.get_emp2(se_a.e_occ, se_a.v_occ, va, 0)
        e_mp2_b += self.get_emp2(se_b.e_occ, se_b.v_occ, vb, 1)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_a, 3)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_b, 3)

    def test_scs_build_dfump2_iter(self):
        (eoa, eob), (eva, evb), (ixq_a, _), (qja_a, qja_b), (axq_a, _), (qbi_a, qbi_b) = aux.dfump2._parse_uhf(self.uhf_df.e, self.eri_df, self.eri_df, self.uhf_df.chempot)
        (eob, eoa), (evb, eva), (ixq_b, _), (qja_b, qja_a), (axq_b, _), (qbi_b, qbi_a) = aux.dfump2._parse_uhf(self.uhf_df.e[::-1], self.eri_df[::-1], self.eri_df[::-1], self.uhf_df.chempot[::-1])
        nocca, noccb, nvira, nvirb = eoa.size, eob.size, eva.size, evb.size
        oa, ob, va, vb = slice(None, nocca), slice(None, noccb), slice(nocca, None), slice(noccb, None)
        se_a_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[0])
        se_b_null = aux.Aux([], [[],]*self.uhf.nao, chempot=self.uhf.chempot[1])
        se_a, se_b = aux.build_dfump2_iter((se_a_null, se_b_null), self.uhf.fock_mo, self.eri_df, wtol=0, os_factor=1.2, ss_factor=0.33)
        e_mp2_a  = self.get_emp2(se_a.e_vir, se_a.v_vir, oa, 0)
        e_mp2_a += self.get_emp2(se_b.e_vir, se_b.v_vir, ob, 1)
        e_mp2_b  = self.get_emp2(se_a.e_occ, se_a.v_occ, va, 0)
        e_mp2_b += self.get_emp2(se_b.e_occ, se_b.v_occ, vb, 1)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_a, 3)
        self.assertAlmostEqual(self.e_mp2_scs, e_mp2_b, 3)


if __name__ == '__main__':
    unittest.main()
