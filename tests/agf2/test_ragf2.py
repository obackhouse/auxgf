import unittest
import numpy as np

from auxgf import mol, hf, aux, agf2


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g')
        self.rhf = hf.RHF(self.m).run()
        self.e_mp2 = -0.04191336686656655

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf

    def test_rmp2(self):
        ragf2 = agf2.RAGF2(self.rhf, nmom=(10, 10), verbose=False)
        self.assertAlmostEqual(ragf2.e_1body, self.rhf.e_tot, 8)
        self.assertAlmostEqual(ragf2.e_2body, self.e_mp2, 8)
        self.assertAlmostEqual(ragf2.e_tot, self.rhf.e_tot + self.e_mp2, 8)
        self.assertAlmostEqual(ragf2.e_mp2, self.e_mp2, 8)

    def test_ragf2_1_1(self):
        ragf2 = agf2.RAGF2(self.rhf, nmom=(1, 1), verbose=False)
        ragf2.run()
        self.assertAlmostEqual(ragf2.e_mp2, self.e_mp2, 3)
        self.assertAlmostEqual(np.trace(ragf2.rdm1), self.m.nelec, 6)
        self.assertAlmostEqual(ragf2.e_1body, -74.92304938877345   , 6)
        self.assertAlmostEqual(ragf2.e_2body, -0.08160143359098293 , 6)
        self.assertAlmostEqual(ragf2.e_hf,    -74.9611711378676    , 6)
        self.assertAlmostEqual(ragf2.e_corr,  -0.043479684496844584, 6)
        self.assertAlmostEqual(ragf2.e_tot,   -75.00465082236444   , 6)

    def test_ragf2_2_2(self):
        ragf2 = agf2.RAGF2(self.rhf, nmom=(1, 1), verbose=False)
        ragf2.run()
        ragf2 = agf2.RAGF2(self.rhf, nmom=(2, 2), dm0=ragf2.rdm1, damping=True, verbose=False)
        ragf2.run()
        self.assertAlmostEqual(ragf2.e_mp2, self.e_mp2, 5)
        self.assertAlmostEqual(np.trace(ragf2.rdm1), self.m.nelec, 6)
        self.assertAlmostEqual(ragf2.e_1body, -74.92013981231922  , 6)
        self.assertAlmostEqual(ragf2.e_2body, -0.08273115876597667, 6)
        self.assertAlmostEqual(ragf2.e_hf,    -74.9611711378676   , 6)
        self.assertAlmostEqual(ragf2.e_corr,  -0.0416998332176064 , 6)
        self.assertAlmostEqual(ragf2.e_tot,   -75.0028709710852   , 6)

    def test_ragf2_None_3(self):
        ragf2 = agf2.RAGF2(self.rhf, nmom=(None, 3), damping=True, delay_damping=5, verbose=False)
        ragf2.run()
        self.assertAlmostEqual(ragf2.e_mp2, self.e_mp2, 6)
        self.assertAlmostEqual(np.trace(ragf2.rdm1), self.m.nelec, 6)
        self.assertAlmostEqual(ragf2.e_1body, -74.92061986016013  , 6)
        self.assertAlmostEqual(ragf2.e_2body, -0.08225785253628955, 6)
        self.assertAlmostEqual(ragf2.e_hf,    -74.9611711378676   , 6)
        self.assertAlmostEqual(ragf2.e_corr,  -0.04170657482882234, 6)
        self.assertAlmostEqual(ragf2.e_tot,   -75.00287771269642  , 6)

    def test_get_fock(self):
        ragf2 = agf2.RAGF2(self.rhf, nmom=(2,2), verbose=False)
        self.assertAlmostEqual(np.max(np.absolute(ragf2.get_fock() - self.rhf.fock_mo)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(ragf2.get_fock(self.rhf.rdm1_mo) - self.rhf.fock_mo)), 0, 8)

    def test_ip(self):
        ragf2 = agf2.RAGF2(self.rhf, nmom=(2, 2), verbose=False)
        ragf2.run()
        w, v = ragf2.se.eig(ragf2.get_fock())
        arg = np.argmax(w[w < ragf2.chempot])
        e1, v1 = -w[w < ragf2.chempot][arg], v[:,w < ragf2.chempot][:,arg][:ragf2.nphys]
        e2, v2 = ragf2.ip
        self.assertAlmostEqual(e1, e2, 8)
        self.assertAlmostEqual(np.linalg.norm(v1), np.linalg.norm(v2), 8)

    def test_ea(self):
        ragf2 = agf2.RAGF2(self.rhf, nmom=(2, 2), verbose=False)
        ragf2.run()
        w, v = ragf2.se.eig(ragf2.get_fock())
        arg = np.argmin(w[w >= ragf2.chempot])
        e1, v1 = w[w >= ragf2.chempot][arg], v[:,w >= ragf2.chempot][:,arg][:ragf2.nphys]
        e2, v2 = ragf2.ea
        self.assertAlmostEqual(e1, e2, 8)
        self.assertAlmostEqual(np.linalg.norm(v1), np.linalg.norm(v2), 8)


if __name__ == '__main__':
    unittest.main()
