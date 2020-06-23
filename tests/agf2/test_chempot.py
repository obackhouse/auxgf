import unittest
import numpy as np

from auxgf import mol, hf, aux, agf2, util


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g')
        self.rhf = hf.RHF(self.m).run()
        self.gf2 = agf2.RAGF2(self.rhf, nmom=(2,3), verbose=False, maxiter=3).run()
        self.se = self.gf2.se
        self.fock = self.gf2.get_fock()

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.gf2, self.se, self.fock

    def test_diag_fock_ext(self):
        w1, v1 = np.linalg.eigh(self.se.as_hamiltonian(self.fock))
        w2, v2, cpt, err = agf2.chempot.diag_fock_ext(self.se, self.fock, self.m.nelec)
        self.assertAlmostEqual(np.max(np.absolute(w1 - w2)), 0, 8)
        self.assertAlmostEqual(np.max(np.absolute(v1 - v2)), 0, 8)

    def test_gradient(self):
        try:
            import warnings
            warnings.simplefilter('ignore', RuntimeWarning)
            import numdifftools as ndt
        except ImportError:
            util.warn('No numdifftools installation detected - skipping gradient test')
            return

        grad = ndt.Gradient(agf2.chempot.objective, step=1e-6, order=3)

        for x in np.linspace(-2.0, 2.0, num=10):
            a = grad(x, self.se, self.fock, self.m.nelec)
            b = agf2.chempot.gradient(x, self.se, self.fock, self.m.nelec)
            self.assertAlmostEqual(a, b, 6)

    def test_brent(self):
        se, opt = agf2.chempot.minimize(self.se, self.fock, self.m.nelec, method='brent', jac=False, bounds=[-1, 1])
        w, v, cpt, err = agf2.chempot.diag_fock_ext(se, self.fock, self.m.nelec)
        self.assertAlmostEqual(err, 0, 5)

    def test_golden(self):
        se, opt = agf2.chempot.minimize(self.se, self.fock, self.m.nelec, method='golden', jac=False, bounds=[-1, 1])
        w, v, cpt, err = agf2.chempot.diag_fock_ext(se, self.fock, self.m.nelec)
        self.assertAlmostEqual(err, 0, 5)

    def test_newton(self):
        se, opt = agf2.chempot.minimize(self.se, self.fock, self.m.nelec, method='newton', jac=False)
        w, v, cpt, err = agf2.chempot.diag_fock_ext(se, self.fock, self.m.nelec)
        self.assertAlmostEqual(err, 0, 5)

    def test_newton_gradient(self):
        se, opt = agf2.chempot.minimize(self.se, self.fock, self.m.nelec, method='newton', jac=True)
        w, v, cpt, err = agf2.chempot.diag_fock_ext(se, self.fock, self.m.nelec)
        self.assertAlmostEqual(err, 0, 5)

    def test_bfgs(self):
        se, opt = agf2.chempot.minimize(self.se, self.fock, self.m.nelec, method='bfgs', jac=False)
        w, v, cpt, err = agf2.chempot.diag_fock_ext(se, self.fock, self.m.nelec)
        self.assertAlmostEqual(err, 0, 5)

    def test_bfgs_gradient(self):
        se, opt = agf2.chempot.minimize(self.se, self.fock, self.m.nelec, method='bfgs', jac=True)
        w, v, cpt, err = agf2.chempot.diag_fock_ext(se, self.fock, self.m.nelec)
        self.assertAlmostEqual(err, 0, 5)

    def test_lstsq(self):
        se, opt = agf2.chempot.minimize(self.se, self.fock, self.m.nelec, method='lstsq', jac=False)
        w, v, cpt, err = agf2.chempot.diag_fock_ext(se, self.fock, self.m.nelec)
        self.assertAlmostEqual(err, 0, 5)

    def test_lstsq_gradient(self):
        se, opt = agf2.chempot.minimize(self.se, self.fock, self.m.nelec, method='lstsq', jac=True)
        w, v, cpt, err = agf2.chempot.diag_fock_ext(se, self.fock, self.m.nelec)
        self.assertAlmostEqual(err, 0, 5)

    def test_stochastic(self):
        se, opt = agf2.chempot.minimize(self.se, self.fock, self.m.nelec, method='stochastic', jac=False, bounds=[-5, 5])
        w, v, cpt, err = agf2.chempot.diag_fock_ext(se, self.fock, self.m.nelec)
        self.assertAlmostEqual(err, 0, 5)


if __name__ == '__main__':
    unittest.main()
