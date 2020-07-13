import unittest
import numpy as np

from auxgf import mol, hf, aux, agf2


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g')
        self.uhf = hf.UHF(self.m).run()
        self.e_mp2 = -0.041913367798116496

    @classmethod
    def tearDownClass(self):
        del self.m, self.uhf

    def test_rmp2(self):
        uagf2 = agf2.UAGF2(self.uhf, nmom=(10, 10), verbose=False)
        self.assertAlmostEqual(uagf2.e_1body, self.uhf.e_tot, 8)
        self.assertAlmostEqual(uagf2.e_2body, self.e_mp2, 8)
        self.assertAlmostEqual(uagf2.e_tot, self.uhf.e_tot + self.e_mp2, 8)
        self.assertAlmostEqual(uagf2.e_mp2, self.e_mp2, 8)

    def test_uagf2_1_1(self):
        uagf2 = agf2.UAGF2(self.uhf, nmom=(1, 1), verbose=False)
        uagf2.run()
        self.assertTrue(uagf2.converged)
        self.assertAlmostEqual(uagf2.e_mp2, self.e_mp2, 3)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[0]), self.m.nalph, 6)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[1]), self.m.nbeta, 6)
        self.assertAlmostEqual(uagf2.e_1body,    -74.9230493877132   , 6)
        self.assertAlmostEqual(uagf2.e_2body,    -0.08160143624651722, 6)
        self.assertAlmostEqual(uagf2.e_hf,       -74.96117113786579  , 6)
        self.assertAlmostEqual(uagf2.e_corr,     -0.04347968609393149, 6)
        self.assertAlmostEqual(uagf2.e_tot,      -75.00465082395972  , 6)

    def test_uagf2_df(self):
        uhf = hf.UHF(self.m, with_df=True).run()
        uagf2 = agf2.UAGF2(uhf, nmom=(1, 1), verbose=False)
        uagf2.run()
        self.assertTrue(uagf2.converged)
        self.assertAlmostEqual(uagf2.e_mp2, self.e_mp2, 3)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[0]), self.m.nalph, 6)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[1]), self.m.nbeta, 6)
        self.assertAlmostEqual(uagf2.e_1body,    -74.9230493877132   , 3)
        self.assertAlmostEqual(uagf2.e_2body,    -0.08160143624651722, 3)
        self.assertAlmostEqual(uagf2.e_hf,       -74.96117113786579  , 3)
        self.assertAlmostEqual(uagf2.e_corr,     -0.04347968609393149, 3)
        self.assertAlmostEqual(uagf2.e_tot,      -75.00465082395972  , 3)

    def test_uagf2_2_2(self):
        uagf2 = agf2.UAGF2(self.uhf, nmom=(1, 1), verbose=False)
        uagf2.run()
        uagf2 = agf2.UAGF2(self.uhf, nmom=(2, 2), dm0=uagf2.rdm1, damping=True, verbose=False)
        uagf2.run()
        self.assertTrue(uagf2.converged)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[0]), self.m.nalph, 6)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[1]), self.m.nbeta, 6)
        self.assertAlmostEqual(uagf2.e_1body,    -74.91982876261184  , 6)
        self.assertAlmostEqual(uagf2.e_2body,    -0.08303716502289221, 6)
        self.assertAlmostEqual(uagf2.e_hf,       -74.96117113786579  , 6)
        self.assertAlmostEqual(uagf2.e_corr,     -0.0416947897689397 , 6)
        self.assertAlmostEqual(uagf2.e_tot,      -75.00286592763473  , 6)

    def test_uagf2_None_3(self):
        uagf2 = agf2.UAGF2(self.uhf, nmom=(None, 3), damping=True, delay_damping=5, verbose=False).run()
        uagf2.run()
        self.assertTrue(uagf2.converged)
        self.assertAlmostEqual(uagf2.e_mp2, self.e_mp2, 6)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[0]), self.m.nalph, 6)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[1]), self.m.nbeta, 6)
        self.assertAlmostEqual(uagf2.e_1body,    -74.92061972014814  , 6)
        self.assertAlmostEqual(uagf2.e_2body,    -0.08271701271449447, 6)
        self.assertAlmostEqual(uagf2.e_hf,       -74.96117113786579  , 6)
        self.assertAlmostEqual(uagf2.e_corr,     -0.04216559499684536, 6)
        self.assertAlmostEqual(uagf2.e_tot,      -75.00333673286264  , 6)

    def test_get_fock(self):
        uagf2 = agf2.UAGF2(self.uhf, nmom=(2,2), verbose=False)
        self.assertAlmostEqual(np.max(np.absolute(uagf2.get_fock() - self.uhf.fock_mo)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(uagf2.get_fock(self.uhf.rdm1_mo) - self.uhf.fock_mo)), 0, 8)

    def test_ip(self):
        uagf2 = agf2.UAGF2(self.uhf, nmom=(2, 2), verbose=False)
        uagf2.run()
        wa, va = uagf2.se[0].eig(uagf2.get_fock()[0])
        wb, vb = uagf2.se[1].eig(uagf2.get_fock()[1])
        arga = np.argmax(wa[wa < uagf2.chempot[0]])
        argb = np.argmax(wb[wb < uagf2.chempot[1]])
        if wa[wa < uagf2.chempot[0]][arga] > wb[wb < uagf2.chempot[1]][argb]:
            e1, v1 = -wa[wa < uagf2.chempot[0]][arga], va[:,wa < uagf2.chempot[0]][:,arga][:uagf2.nphys]
        else:
            e1, v1 = -wb[wb < uagf2.chempot[1]][argb], vb[:,wb < uagf2.chempot[1]][:,argb][:uagf2.nphys]
        e2, v2 = uagf2.ip
        self.assertAlmostEqual(e1, e2, 8)
        self.assertAlmostEqual(np.linalg.norm(v1), np.linalg.norm(v2), 8)

    def test_ea(self):
        uagf2 = agf2.UAGF2(self.uhf, nmom=(2, 2), verbose=False)
        uagf2.run()
        wa, va = uagf2.se[0].eig(uagf2.get_fock()[0])
        wb, vb = uagf2.se[1].eig(uagf2.get_fock()[1])
        arga = np.argmin(wa[wa >= uagf2.chempot[0]])
        argb = np.argmin(wb[wb >= uagf2.chempot[1]])
        if wa[wa >= uagf2.chempot[0]][arga] > wb[wb >= uagf2.chempot[1]][argb]:
            e1, v1 = wa[wa >= uagf2.chempot[0]][arga], va[:,wa >= uagf2.chempot[0]][:,arga][:uagf2.nphys]
        else:
            e1, v1 = wb[wb >= uagf2.chempot[1]][argb], vb[:,wb >= uagf2.chempot[1]][:,argb][:uagf2.nphys]
        e2, v2 = uagf2.ea
        self.assertAlmostEqual(e1, e2, 8)
        self.assertAlmostEqual(np.linalg.norm(v1), np.linalg.norm(v2), 8)


if __name__ == '__main__':
    unittest.main()
