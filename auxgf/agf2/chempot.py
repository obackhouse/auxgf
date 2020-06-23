''' Functions to optimise the chemical potential of the auxiliaries.
'''

import numpy as np
from scipy import optimize

from auxgf import util
from auxgf.util import types, log, mpi


def diag_fock_ext(se, fock, nelec, chempot=0.0, occupancy=2.0, buf=None):
    ''' Diagonalises the extended Fock matrix and returns finds a
        chemical potential via Aufbau.

    Parameters
    ----------
    se : Aux
        auxiliaries
    fock : ndarray
        physical space Fock matrix
    nelec : int
        target number of electrons
    chempot : float, optional
        chemical potential on the auxiliary space (NOT the same as
        that on the physical space!), default 0.0
    occupancy : float, optional
        orbital occupancy, i.e. 2 for RHF and 1 for UHF, default 2
    buf : ndarray, optional
        array to store the extended Fock matrix
    
    Returns
    -------
    w : ndarray
        eigenvalues
    v : ndarray
        eigenvectors
    chempot : float
        chemical potential on the physical space which best satisfies
        the number of electrons
    error : float
        error in the number of electrons in the physical space
    '''

    f_ext = se.as_hamiltonian(fock, chempot=chempot, out=buf)
    w, v = util.eigh(f_ext)

    chempot_phys, error = util.chempot.find_chempot(se.nphys, nelec, h=(w,v),
                                                    occupancy=occupancy)

    return w, v, chempot_phys, error


def objective(x, se, fock, nelec, occupancy=2.0, buf=None):
    ''' Objective function for the minimization, i.e. f(chempot) = error
        in the diag_fock_ext function.

    Parameters
    ----------
    x : float
        objective parameter, i.e. chemical potential on the auxiliary
        space
    buf : ndarray, optional
        array to store the extended Fock matrix
    occupancy : float, optional
        orbital occupancy, i.e. 2 for RHF and 1 for UHF, default 2

    Returns
    -------
    fx : float
        objective function value, i.e. error in number of electrons
    '''

    w, v, chempot, error = diag_fock_ext(se, fock, nelec, chempot=x, buf=buf,
                                         occupancy=occupancy)

    return error**2


def gradient(x, se, fock, nelec, occupancy=2.0, buf=None, return_val=False):
    ''' Gradient function for the minimization with respect to error.

    Parameters
    ----------
    x : float
        objective parameter, i.e. chemical potential on the auxiliary
        space
    buf : ndarray, optional
        array to store the extended Fock matrix
    occupancy : float, optional
        orbital occupancy, i.e. 2 for RHF and 1 for UHF, default 2

    Returns
    -------
    fx : float
        objective function value, i.e. error in number of electrons
    dx : float
        gradient value
    '''

    w, v, chempot, error = diag_fock_ext(se, fock, nelec, chempot=x, buf=buf,
                                         occupancy=occupancy)

    nocc = np.sum(w < chempot)
    phys = slice(None, se.nphys)
    aux = slice(se.nphys, None)
    occ = slice(None, nocc)
    vir = slice(nocc, None)

    h_1 = -np.dot(v[aux,vir].T, v[aux,occ])
    z_ai = -h_1 / util.dirsum('i,a->ai', w[occ], -w[vir])
    c_occ = np.dot(v[phys,vir], z_ai)

    rdm1 = np.dot(v[phys,occ], c_occ.T) * 4
    ne = np.trace(rdm1)

    d = 2 * error * ne

    if return_val:
        return error**2, d
    else:
        return d


def minimize(se, fock, nelec, occupancy=2.0, method='newton', buf=None, x0=0.0, bounds=(None, None), tol=1e-6, maxiter=200, jac=True):
    ''' Finds a set of auxiliary energies and chemical potential on
        the physical space which best satisfies then number of 
        electrons.

        If method is brent, golden or stochastic, then bounds argument
        should be passed. However, using a method with x0 is advised.

    Parameters
    ----------
    se : Aux
        auxiliaries
    fock : ndarray
        physical space Fock matrix
    nelec : int
        target number of electrons
    occupancy : float, optional
        orbital occupancy, i.e. 2 for RHF and 1 for UHF, default 2
    buf : ndarray, optional
        array to store the extended Fock matrix
    method : str, optional
        minimization algorithm to use, default 'slsqp'
    x0 : float, optional
        initial guess, default 0.0
    bounds : tuple of floats, optional
        bounds for solution
    tol : float, optional
        tolerance for convergence (in no. of electrons), default 1e-6
    maxiter : int, optional
        maximum number of iterations, default 200
    jac : bool, optional
        if True, use the gradient function, default True

    Returns
    -------
    se : Aux
        auxiliaries, with changed energies and chemical potential
    opt : OptimizeResult
        SciPy optimization result object

    Raises
    ------
    ValueError
        if method is one of brent, golden or stochastic, and bounds is
        not provided
    ValueError
        when an incompatible method and jac are provided
    '''

    if jac is True and method not in ['newton', 'bfgs', 'lstsq']:
        raise ValueError('method=%s and jac=%s not supported' % (method, jac))

    if bounds is (None, None) and method in ['brent', 'golden', 'stochastic']:
        raise ValueError('method=%s requires bounds argument' % method)

    if bounds != (None, None) and (x0 < bounds[0] or x0 > bounds[1]):
        x0 = 0.5 * (bounds[0] + bounds[1])

    tol = tol ** 2

    if method == 'brent':
        f = optimize.minimize_scalar
        options = dict(maxiter=maxiter, xtol=tol)
        kwargs = dict(method='brent', bounds=bounds, options=options)

    elif method == 'golden':
        f = optimize.minimize_scalar
        options = dict(maxiter=maxiter, xtol=tol)
        kwargs = dict(method='golden', bounds=bounds, options=options)

    elif method == 'newton':
        f = optimize.minimize
        options = dict(maxiter=maxiter, ftol=tol, xtol=tol, gtol=tol)
        kwargs = dict(x0=x0, method='TNC', bounds=[bounds], options=options)
        if jac:
            kwargs['jac'] = lambda *args: gradient(*args)

    elif method == 'bfgs':
        f = optimize.minimize
        options = dict(maxiter=maxiter, ftol=tol)
        kwargs = dict(x0=x0, method='L-BFGS-B', bounds=[bounds], options=options)
        if jac:
            kwargs['jac'] = lambda *args: np.array([gradient(*args)])

    elif method == 'lstsq':
        f = optimize.minimize
        options = dict(maxiter=maxiter, ftol=tol)
        kwargs = dict(x0=x0, method='SLSQP', options=options)
        if jac:
            kwargs['jac'] = lambda *args: gradient(*args)

    elif method == 'stochastic':
        f = optimize.differential_evolution
        kwargs = dict(bounds=[bounds], maxiter=maxiter, tol=tol)

    else:
        raise ValueError

    args = (se, fock, nelec, occupancy, buf)
    opt = f(objective, args=args, **kwargs)

    se._ener -= opt.x
    se.chempot = util.chempot.find_chempot(se.nphys, nelec, h=se.eig(fock))

    return se, opt
