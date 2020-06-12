import unittest
import numpy as np

from auxgf import mol, hf, aux, grids


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g')
        self.rhf = hf.RHF(self.m).run()
        self.se = aux.build_rmp2(self.rhf.e, self.rhf.eri_mo, chempot=self.rhf.chempot)
        self.imfq = grids.ImFqGrid(2**5, beta=2**3)
        self.refq = grids.ReFqGrid(2**5, minpt=-5, maxpt=5, eta=0.01)
        self.imqd = grids.ImFqQuad(2**5, beta=2**3, lamb=0.01)

    @classmethod
    def tearDownClass(self):
        del self.m, self.rhf, self.se, self.imfq, self.refq, self.imqd

    def test_grad(self):
        import numdifftools as ndt

        fit = aux.fit.FitHelper(self.se, self.imfq, hessian=False)
        obj = lambda *args : aux.fit.function(*args)
        grad_ref = ndt.Gradient(obj, x, fit)
        grad = aux.fit.objective(x, fit)[1]

        mask = np.isclose(grad, grad_ref, rtol=1e-6, atol=1e-8)
        e, v = fit.unpack(mask)

        self.assertTrue(np.all(e))
        self.assertTrue(np.all(v))

    def test_hess(self):
        import numdifftools as ndt

        fit = aux.fit.FitHelper(self.se, self.imfq, hessian=False)
        obj = lambda *args : aux.fit.function(*args)
        hess_ref = ndt.Hessian(obj, step=1e-5)(x, fit)
        hess = hessian(x, fit)

        mask = np.isclose(hess, hess_ndt, rtol=1e-6, atol=1e-8)
        ee = mask[fit.se,fit.se]
        ev = mask[fit.se,fit.sv]
        ve = mask[fit.sv,fit.se]
        vv = mask[fit.sv,fit.sv]

        self.assertTrue(np.all(ee))
        self.assertTrue(np.all(ev))
        self.assertTrue(np.all(ve))
        self.assertTrue(np.all(vv))

    def test_imfq(self):
        se_guess = aux.Aux(self.rhf.e, np.eye(self.rhf.nao))
        se_fit = aux.fit.run(se_guess, self.se.as_spectrum(self.imfq), self.imfq, hessian=False)
        print(np.linalg.norm(se_guess.as_spectrum(self.imfq), se_fit.as_spectrum(self.imfq)))


if __name__ == '__main__':
    unittest.main()
