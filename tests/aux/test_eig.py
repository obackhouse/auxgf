import unittest
import numpy as np

from auxgf import util, mol, hf, aux


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g')
        self.rhf = hf.RHF(self.m).run()
        self.fock = self.rhf.fock_mo
        self.se = aux.build_rmp2(self.rhf.e, self.rhf.eri_mo, chempot=self.rhf.chempot) #FIXME remove test dependency?
        self.f_ext = self.se.as_hamiltonian(self.fock)

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.fock, self.se

    def test_eigh(self):
        w, v = aux.eig.dense(self.se, self.fock)
        f_ext = np.einsum('xk,k,yk->xy', v, w, v)
        self.assertAlmostEqual(np.max(np.absolute(f_ext - self.f_ext)), 0, 8)

    def test_davidson(self):
        w1, v1 = aux.eig.dense(self.se, self.fock)
        w2, v2 = aux.eig.davidson(self.se, self.fock, nroots=1, which='SA')
        w3, v3 = aux.eig.davidson(self.se, self.fock, nroots=1, which='LA')
        w4, v4 = aux.eig.davidson(self.se, self.fock, nroots=1, which='SM')
        w5, v5 = aux.eig.davidson(self.se, self.fock, nroots=1, which='LM')
        self.assertAlmostEqual(w1[np.argmin(w1)] - w2[0], 0, 8)
        self.assertAlmostEqual(w1[np.argmax(w1)] - w3[0], 0, 8)
        self.assertAlmostEqual(w1[np.argmin(np.absolute(w1))] - w4[0], 0, 8)
        self.assertAlmostEqual(w1[np.argmax(np.absolute(w1))] - w5[0], 0, 8)

    def test_lanczos(self):
        w1, v1 = aux.eig.dense(self.se, self.fock)
        w2, v2 = aux.eig.lanczos(self.se, self.fock, nroots=1, which='SA')
        w3, v3 = aux.eig.lanczos(self.se, self.fock, nroots=1, which='LA')
        w4, v4 = aux.eig.lanczos(self.se, self.fock, nroots=1, which='SM')
        w5, v5 = aux.eig.lanczos(self.se, self.fock, nroots=1, which='LM')
        self.assertAlmostEqual(w1[np.argmin(w1)] - w2[0], 0, 8)
        self.assertAlmostEqual(w1[np.argmax(w1)] - w3[0], 0, 8)
        self.assertAlmostEqual(w1[np.argmin(np.absolute(w1))] - w4[0], 0, 8)
        self.assertAlmostEqual(w1[np.argmax(np.absolute(w1))] - w5[0], 0, 8)


if __name__ == '__main__':
    unittest.main()
