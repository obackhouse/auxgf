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
        self.assertAlmostEqual(uagf2.e_mp2, self.e_mp2, 3)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[0]), self.m.nalph)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[1]), self.m.nbeta)
        self.assertAlmostEqual(uagf2.e_1body,    -74.9230493877132   , 8)
        self.assertAlmostEqual(uagf2.e_2body,    -0.08160143624651722, 8)
        self.assertAlmostEqual(uagf2.e_hf,       -74.96117113786579  , 8)
        self.assertAlmostEqual(uagf2.e_corr,     -0.04347968609393149, 8)
        self.assertAlmostEqual(uagf2.e_tot,      -75.00465082395972  , 8)
        self.assertAlmostEqual(uagf2.chempot[0], 0.12835886011811576 , 8)
        self.assertAlmostEqual(uagf2.chempot[1], 0.12835886011818873 , 8)

    def test_uagf2_2_2(self):
        uagf2 = agf2.UAGF2(self.uhf, nmom=(1, 1), verbose=False)
        uagf2.run()
        uagf2 = agf2.UAGF2(self.uhf, nmom=(2, 2), dm0=uagf2.rdm1, damping=True, verbose=False)
        uagf2.run()
        self.assertAlmostEqual(uagf2.e_mp2, self.e_mp2, 5)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[0]), self.m.nalph)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[1]), self.m.nbeta)
        self.assertAlmostEqual(uagf2.e_1body,    -74.92014093105244  , 8)
        self.assertAlmostEqual(uagf2.e_2body,    -0.0827314167951274 , 8)
        self.assertAlmostEqual(uagf2.e_hf,       -74.96117113786579  , 8)
        self.assertAlmostEqual(uagf2.e_corr,     -0.04170120998178106, 8)
        self.assertAlmostEqual(uagf2.e_tot,      -75.00287234784757  , 8)
        self.assertAlmostEqual(uagf2.chempot[0], 0.1332917465325183  , 8)
        self.assertAlmostEqual(uagf2.chempot[1], 0.13329122995866868 , 8)

    def test_uagf2_None_3(self):
        uagf2 = agf2.UAGF2(self.uhf, nmom=(None, 3), damping=True, delay_damping=5, verbose=False).run()
        uagf2.run()
        self.assertAlmostEqual(uagf2.e_mp2, self.e_mp2, 6)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[0]), self.m.nalph)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[1]), self.m.nbeta)
        self.assertAlmostEqual(uagf2.e_1body,    -74.92061972014814  , 8)
        self.assertAlmostEqual(uagf2.e_2body,    -0.08271701271449447, 8)
        self.assertAlmostEqual(uagf2.e_hf,       -74.96117113786579  , 8)
        self.assertAlmostEqual(uagf2.e_corr,     -0.04216559499684536, 8)
        self.assertAlmostEqual(uagf2.e_tot,      -75.00333673286264  , 8)
        self.assertAlmostEqual(uagf2.chempot[0], 0.13491238500738034 , 8)
        self.assertAlmostEqual(uagf2.chempot[1], 0.1349123850102253  , 8)

    def test_get_fock(self):
        uagf2 = agf2.UAGF2(self.uhf, nmom=(2,2), verbose=False)
        self.assertAlmostEqual(np.max(np.absolute(uagf2.get_fock() - self.uhf.fock_mo)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(uagf2.get_fock(self.uhf.rdm1_mo) - self.uhf.fock_mo)), 0, 8)


if __name__ == '__main__':
    unittest.main()
