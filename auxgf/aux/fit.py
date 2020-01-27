''' Routines to fit sets of auxiliaries to a frequency-dependent
    spectral function.
'''

import numpy as np
import scipy.optimize

from auxgf import util
from auxgf.util import types


class FitHelper:
    ''' Helper class for fitting procedure.

    Parameters
    ----------
    aux : Aux
        auxiliaries
    grid : (n) ImFqGrid, ImFqQuad or ReFqGrid
        frequency grid
    target : (n,m,m) ndarray
        fitting target, as a spectrum expressed on `grid`
    hessian : bool, optional
        if True, use the analytic Hessian to fit function

    Attributes
    ----------
    ncoup : int
        number of couplings (nphys * naux)
    nvar : int
        number of fitting variables (naux + ncoup)
    se : slice
        slice corresponding to energies
    sv : slice
        slice corresponding to couplinhd
    x0 : (n) ndarray
        initial vector of fitting variables

    Methods
    -------
    pack(e, v)
        packs a set of auxiliary energies and couplings into a vector
    unpack(x)
        unpacks a vector into a set of auxiliary energies and couplings
    generate(x)
        calls `unpack` and also returns the spectral function, and the
        difference thereof with respect to the target function
    build_spectrum(e, v, ordering='feynman')
        builds the spectrum using auxiliary energies `e` and couplings
        `v` on the frequency grid `grid`
    build_denominator(e, v, ordering='feynman')
        builds the denominator array requrired to build the spectrum
    '''

    def __init__(self, aux, grid, target, hessian=True):
        self.nphys, self.naux = aux.v.shape
        self.ncoup = aux.v.size
        self.nvar = self.ncoup + self.naux

        self.se = slice(None, self.naux)
        self.sv = slice(self.naux, None)

        self.x0 = np.block([aux.e, aux.v.flatten()])

        self.grid = grid

        if grid.wts is None:
            self.wts = np.ones((grid.npts,), dtype=types.float64)
            self.wts /= grid.npts
        else:
            self.wts = grid.wts.copy()

        self.target = target
        self.chempot = chempot
        self.hessian = hessian

    def pack(self, e, v):
        return np.block([e, v.flatten()])

    def unpack(self, x):
        e = x[self.se]
        v = x[self.sv]

        v = v.reshape((self.nphys, self.naux))

        return e, v

    def generate(self, x):
        e, v = self.unpack(x)

        sf = self.build_spectrum(e, v)
        dsf = sf - self.target
        
        return e, v, sf, dsf

    def build_spectrum(self, e, v, ordering='feynman'):
        return self.aux.build_spectrum(self.grid, e, v, chempot=self.chempot, 
                                       ordering=ordering)

    def build_denominator(self, e, v, ordering='feynman'):
        return self.aux.build_denominator(self.grid, e, v, chempot=self.chempot,
                                          ordering=ordering)


def function(x, fit):
    ''' Computes the target function.

    Parameters
    ----------
    x : (n) ndarray
        vector of fitting variables
    fit : FitHelper
        helper class

    Returns
    -------
    f : float
        sum of f(x)
    '''

    e, v, sf, dsf = fit.generate(x)

    f  = util.einsum('w,wxy->', fit.wts, dsf.real**2)
    f += util.einsum('w,wxy->', fit.wts, dsf.imag**2)

    return f


def objective(x, fit):
    ''' Computes the objective function and gradient.

    Parameters
    ----------
    x : (n) ndarray
        vector of fitting variables
    fit : FitHelper
        helper class

    Returns
    -------
    f : float
        sum of f(x)
    dx : (n) ndarray
        vector of the derivative of f(x)
    '''

    e, v, sf, dsf = fit.generate(x)

    r = fit.build_denominator(e, v)

    gv = util.einsum('xk,wk->wxk', v, r)
    ge = util.einsum('wxk,wyk->wxyk', gv, gv)

    f  = util.einsum('w,wxy->', fit.wts, dsf.real**2)
    f += util.einsum('w,wxy->', fit.wts, dsf.imag**2)

    de  = util.einsum('w,wxy,wxyk->k', fit.wts, dsf.real, ge.real)
    de += util.einsum('w,wxy,wxyk->k', fit.wts, dsf.imag, ge.imag)
    de *= 2

    dv  = util.einsum('w,wzx,wxk->zk', fit.wts, dsf.real, gv.real)
    dv += util.einsum('w,wzx,wxk->zk', fit.wts, dsf.imag, ge.imag)
    dv *= 4

    return f, fit.pack(de, dv)


def hessian(x, fit):
    ''' Computes the Hessian of the function.

    Parameters
    ----------
    x : (n) ndarray
        vector of fitting variables
    fit : FitHelper
        helper class

    Returns
    -------
    hx : (n,n) ndarray
        matrix of the Hessian of f(x)
    '''

    e, v, sf, dsf = fit.generate(x)

    r = fit.build_denominator(e, v)

    gv = util.einsum('xk,wk->wxk', v, r)
    ge = util.einsum('wxk,wyk->wxyk', gv, gv)

    gee = util.einsum('wk,wxyk->wxyk', r, ge) * 2
    gev = util.einsum('wk,wxk->wxk', r, gv)
    gvv = r

    hee  = util.einsum('w,wxyk,wxyl->kl', fit.wts, ge.real, ge.real)
    hee += util.einsum('w,wxyk,wxyl->kl', fit.wts, ge.imag, ge.imag)
    hee *= 2

    dee  = util.einsum('w,wxy,wxyk->k', fit.wts, dsf.real, gee.real)
    dee += util.einsum('w,wxy,wxyk->k', fit.wts, dsf.imag, gee.imag)
    dee *= 2
    
    diag = util.diagonal(hee)
    diag += dee

    hev  = util.einsum('w,wxzk,wxl->kzl', fit.wts, ge.real, gv.real)
    hev += util.einsum('w,wxzk,wxl->kzl', fit.wts, ge.imag, gv.imag)
    hev *= 4

    dev  = util.einsum('w,wxz,wxl->zl', fit.wts, dsf.real, gev.real)
    dev += util.einsum('w,wxz,wxl->zl', fit.wts, dsf.imag, gev.imag)
    dev *= 4

    diag = util.diagonal(hev, axis1=0, axis2=2)
    diag += dev

    hev = hev.reshape((fit.naux, fit.ncoup))

    hvv  = util.einsum('w,wyk,wxl->xkyl', fit.wts, gv.real, gv.real)
    hvv += util.einsum('w,wyk,wxl->xkyl', fit.wts, gv.imag, gv.imag)
    hvv *= 4

    dvv  = util.einsum('w,wxk,wxl->kl', fit.wts, gv.real, gv.real)
    dvv += util.einsum('w,wxk,wxl->kl', fit.wts, gv.imag, gv.imag)
    dvv *= 4

    diag = util.diagonal(hvv, axis1=0, axis2=2)
    diag += dvv[:,:,None]

    dvv  = util.einsum('w,wxy,wl->xyl', fit.wts, dsf.real, gvv.real)
    dvv += util.einsum('w,wxy,wl->xyl', fit.wts, dsf.imag, gvv.imag)
    dvv *= 4

    diag = util.diagonal(hvv, axis1=1, axis2=3)
    diag += dvv

    hvv = hvv.reshape((fit.ncoup, fit.ncoup))

    h = np.block([[hee, hev], [hev.T, hvv]])

    return h


def test_gradient(x, fit, rtol=1e-6, atol=1e-8):
    import numdifftools as ndt

    obj = lambda *args: function(*args)
    grad_ndt = ndt.Gradient(obj, x, fit)
    grad = objective(x, fit)[1]

    mask = np.isclose(grad, grad_ndt, rtol=rtol, atol=atol)
    em, vm = fit.unpack(mask)

    assert np.all(em)
    assert np.all(vm)


def test_hessian(x, fit, rtol=1e-6, atol=1e-8):
    import numdifftools as ndt

    obj = lambda *args: function(*args)
    hess_ndt = ndt.Hessian(obj, step=1e-5)(x, fit)
    hess = hessian(x, fit)

    mask = np.isclose(hess, hess_ndt, rtol=rtol, atol=atol)
    eem = mask[fit.se,fit.se]
    evm = mask[fit.se,fit.sv]
    vem = mask[fit.sv,fit.se]
    vvm = mask[fit.sv,fit.sv]

    assert np.all(eem)
    assert np.all(evm)
    assert np.all(vem)
    assert np.all(vvm)


def run(aux, target, grid, hessian=True, opts={}, test_grad=False, test_hess=False):
    ''' Runs the auxiliary fitting procedure. Trust-exact is used when
        `hessian` is True, otherwise BFGS is used.

    Parameters
    ----------
    aux : Aux
        initial auxiliaries
    target : (n,m,m) ndarray
        target spectral function i.e. `aux` expressed on `grid`
    grid : (k) ImFqGrid, ImFqQuad or ReFqGrid
        frequency grid
    hessian : bool, optional
        if True, use the analytic Hessian to fit function
    opts : dict, optional
        additional fitting options for SciPy optimizer
    test_grad : bool, optional
        if True, test the gradient against a numerical result using
        numdifftools, default False
    test_hessian : bool, optional
        if True, test the hessian against a numerical result using
        numdifftools, default False

    Returns
    -------
    aux_fit : Aux
        fitted auxiliaries
    res : scipy.optimize.OptimizeResult
        result of SciPy optimization
    '''

    fit = FitHelper(aux, grid, target, hessian=hessian, opts=opts)

    opts['maxiter'] = opts.get('maxiter', 10000)
    opts['gtol'] = opts.get('gtol', 1e-8)

    if test_grad:
        test_gradient(fit.x0, fit)

    if test_hess:
        test_hessian(fit.x0, fit)

    args = { 'x0': fit.x0, 'args': (fit,), 'jac': True, 'hess': hessian, 
             'options': opts, 'method': 'trust-exact' if hessian else 'BFGS' }

    res = scipy.optimize.minimize(objective, *args)

    e, v = fit.unpack(res.x)

    aux_fit = aux.new(e, v)

    return aux_fit, res
























