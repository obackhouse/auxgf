import unittest
import numpy as np

from auxgf import mol, hf, aux, agf2, mpi


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g')
        self.uhf = hf.UHF(self.m).run()
        self.uhf_df = hf.UHF(self.m, auxbasis='aug-cc-pvqz-ri', with_df=True).run()
        self.e_mp2 = -0.041913367798116496

    @classmethod
    def tearDownClass(self):
        del self.m, self.uhf, self.uhf_df, self.e_mp2

    def test_ump2(self):
        uagf2 = mpi.agf2.OptUAGF2(self.uhf_df, verbose=False)
        self.assertAlmostEqual(uagf2.e_1body, self.uhf_df.e_tot, 8)
        self.assertAlmostEqual(uagf2.e_2body, self.e_mp2, 2)
        self.assertAlmostEqual(uagf2.e_tot, self.uhf_df.e_tot + self.e_mp2, 2)
        self.assertAlmostEqual(uagf2.e_mp2, self.e_mp2, 2)

    def test_uagf2(self):
        # Dependent upon UAGF2 passing tests
        dm0 = self.uhf.rdm1_mo
        opt_uagf2 = mpi.agf2.OptUAGF2(self.uhf_df, dm0=dm0, verbose=False, etol=1e-7)
        opt_uagf2.run()
        uagf2 = agf2.UAGF2(self.uhf, nmom=(None,0), verbose=False, etol=1e-7)
        uagf2.run()
        self.assertAlmostEqual(uagf2.e_mp2, opt_uagf2.e_mp2, 4)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[0]), np.trace(opt_uagf2.rdm1[0]), 4)
        self.assertAlmostEqual(np.trace(uagf2.rdm1[1]), np.trace(opt_uagf2.rdm1[1]), 4)
        self.assertAlmostEqual(uagf2.e_1body, opt_uagf2.e_1body, 3)
        self.assertAlmostEqual(uagf2.e_2body, opt_uagf2.e_2body, 3)
        self.assertAlmostEqual(uagf2.e_hf, opt_uagf2.e_hf, 3)
        self.assertAlmostEqual(uagf2.e_corr, opt_uagf2.e_corr, 3)
        self.assertAlmostEqual(uagf2.e_tot, opt_uagf2.e_tot, 3)
        self.assertAlmostEqual(uagf2.chempot[0], opt_uagf2.chempot[0], 4)
        self.assertAlmostEqual(uagf2.chempot[1], opt_uagf2.chempot[1], 4)

    def test_ip(self):
        opt_uagf2 = mpi.agf2.OptUAGF2(self.uhf_df, verbose=False, etol=1e-7)
        opt_uagf2.run()
        wa, va = opt_uagf2.se[0].eig(opt_uagf2.get_fock()[0])
        wb, vb = opt_uagf2.se[1].eig(opt_uagf2.get_fock()[1])
        opt_uagf2.gf = (opt_uagf2.gf[0].new(wa, va[:opt_uagf2.nphys]), opt_uagf2.gf[1].new(wb, vb[:opt_uagf2.nphys]))
        uagf2 = agf2.UAGF2(self.uhf, nmom=(None,0), verbose=False, etol=1e-7)
        uagf2.run()
        wa, va = opt_uagf2.se[0].eig(opt_uagf2.get_fock()[0])
        wb, vb = opt_uagf2.se[1].eig(opt_uagf2.get_fock()[1])
        arga = np.argmax(wa[wa < opt_uagf2.chempot[0]])
        argb = np.argmax(wb[wb < opt_uagf2.chempot[1]])
        e1, v1 = opt_uagf2.ip
        e2, v2 = uagf2.ip
        self.assertAlmostEqual(e1, e2, 4)
        self.assertAlmostEqual(np.linalg.norm(v1), np.linalg.norm(v2), 4)

    def test_ea(self):
        opt_uagf2 = mpi.agf2.OptUAGF2(self.uhf_df, verbose=False, etol=1e-7)
        opt_uagf2.run()
        wa, va = opt_uagf2.se[0].eig(opt_uagf2.get_fock()[0])
        wb, vb = opt_uagf2.se[1].eig(opt_uagf2.get_fock()[1])
        opt_uagf2.gf = (opt_uagf2.gf[0].new(wa, va[:opt_uagf2.nphys]), opt_uagf2.gf[1].new(wb, vb[:opt_uagf2.nphys]))
        uagf2 = agf2.UAGF2(self.uhf, nmom=(None,0), verbose=False, etol=1e-7)
        uagf2.run()
        wa, va = opt_uagf2.se[0].eig(opt_uagf2.get_fock()[0])
        wb, vb = opt_uagf2.se[1].eig(opt_uagf2.get_fock()[1])
        arga = np.argmin(wa[wa >= opt_uagf2.chempot[0]])
        argb = np.argmin(wb[wb >= opt_uagf2.chempot[1]])
        e1, v1 = opt_uagf2.ea
        e2, v2 = uagf2.ea
        self.assertAlmostEqual(e1, e2, 4)
        self.assertAlmostEqual(np.linalg.norm(v1), np.linalg.norm(v2), 4)


if __name__ == '__main__':
    unittest.main()
