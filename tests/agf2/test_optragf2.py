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
        self.rhf_df = hf.RHF(self.m, auxbasis='aug-cc-pvqz-ri', with_df=True).run()
        self.e_mp2 = -0.041920665444464156

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.rhf_df, self.e_mp2

    def test_rmp2(self):
        ragf2 = agf2.OptRAGF2(self.rhf_df, verbose=False)
        self.assertAlmostEqual(ragf2.e_1body, self.rhf_df.e_tot, 8)
        self.assertAlmostEqual(ragf2.e_2body, self.e_mp2, 2)
        self.assertAlmostEqual(ragf2.e_tot, self.rhf_df.e_tot + self.e_mp2, 2)
        self.assertAlmostEqual(ragf2.e_mp2, self.e_mp2, 2)

    def test_ragf2(self):
        # Dependent upon RAGF2 passing tests
        dm0 = self.rhf.rdm1_mo
        opt_ragf2 = agf2.OptRAGF2(self.rhf_df, dm0=dm0, verbose=False, etol=1e-7)
        opt_ragf2.run()
        ragf2 = agf2.RAGF2(self.rhf, nmom=(None,0), verbose=False, etol=1e-7)
        ragf2.run()
        self.assertAlmostEqual(ragf2.e_mp2, opt_ragf2.e_mp2, 4)
        self.assertAlmostEqual(np.trace(ragf2.rdm1), np.trace(opt_ragf2.rdm1), 4)
        self.assertAlmostEqual(ragf2.e_1body, opt_ragf2.e_1body, 3)
        self.assertAlmostEqual(ragf2.e_2body, opt_ragf2.e_2body, 3)
        self.assertAlmostEqual(ragf2.e_hf, opt_ragf2.e_hf, 3)
        self.assertAlmostEqual(ragf2.e_corr, opt_ragf2.e_corr, 3)
        self.assertAlmostEqual(ragf2.e_tot, opt_ragf2.e_tot, 3)
        self.assertAlmostEqual(ragf2.chempot, opt_ragf2.chempot, 4)

    def test_ip(self):
        opt_ragf2 = agf2.OptRAGF2(self.rhf_df, verbose=False, etol=1e-7)
        opt_ragf2.run()
        ragf2 = agf2.RAGF2(self.rhf, nmom=(None,0), verbose=False, etol=1e-7)
        ragf2.run()
        w, v = opt_ragf2.se.eig(opt_ragf2.get_fock())
        arg = np.argmax(w[w < opt_ragf2.chempot])
        e1, v1 = -w[w < opt_ragf2.chempot][arg], v[:,w < opt_ragf2.chempot][:,arg][:opt_ragf2.nphys]
        e2, v2 = opt_ragf2.ip
        e3, v3 = ragf2.ip
        self.assertAlmostEqual(e1, e2, 8)
        self.assertAlmostEqual(e2, e3, 4)
        self.assertAlmostEqual(np.linalg.norm(v1), np.linalg.norm(v2), 8)
        self.assertAlmostEqual(np.linalg.norm(v2), np.linalg.norm(v3), 4)

    def test_ea(self):
        opt_ragf2 = agf2.OptRAGF2(self.rhf_df, verbose=False, etol=1e-7)
        opt_ragf2.run()
        ragf2 = agf2.RAGF2(self.rhf, nmom=(None,0), verbose=False, etol=1e-7)
        ragf2.run()
        w, v = opt_ragf2.se.eig(opt_ragf2.get_fock())
        arg = np.argmin(w[w >= opt_ragf2.chempot])
        e1, v1 = w[w >= opt_ragf2.chempot][arg], v[:,w >= opt_ragf2.chempot][:,arg][:opt_ragf2.nphys]
        e2, v2 = opt_ragf2.ea
        e3, v3 = ragf2.ea
        self.assertAlmostEqual(e1, e2, 8)
        self.assertAlmostEqual(e2, e3, 4)
        self.assertAlmostEqual(np.linalg.norm(v1), np.linalg.norm(v2), 8)
        self.assertAlmostEqual(np.linalg.norm(v2), np.linalg.norm(v3), 4)


if __name__ == '__main__':
    unittest.main()
