''' Functions to perform the Fock loop used in auxiliary GF2.
'''

import numpy as np

from auxgf import util
from auxgf.util import types, log, mpi
from auxgf.agf2.chempot import minimize, diag_fock_ext

# If anyone is reading this I sincerely apologise for how shit this code is


def _set_options(**kwargs):
    options = { 'fock_func': None,
                'diis_space': 8,
                'netol': 1e-6,
                'dtol': 1e-6,
                'verbose': True,
                'maxiter': 50,
                'maxruns': 20,
                'frozen': 0,
                'method': 'newton',
                'jac': True,
    }

    for key,val in kwargs.items():
        if key not in options.keys():
            raise ValueError('%s argument invalid.' % key)

    options.update(kwargs)

    options['verbose'] = options['verbose'] and not mpi.rank

    return options


def _get_fock_ext(fock, e, v, chempot):
    return np.block([[fock, v], [v.T, np.diag(e-chempot)]])


def fock_loop_rhf(se, hf, rdm, **kwargs):
    ''' Performs the self-consistent loop of the Fock matrix in
        auxiliary GF2 for restricted wavefunctions.

    Parameters
    ----------
    se : Aux
        object containing the self-energy auxiliary space
    hf : hf.RHF object
        Hartree-Fock object
    rdm : (n,n) ndarray
        initial reduced density matrix
    fock_func : callable, optional
        function to return the fock matrix, taking rdm as an argument
    diis_space : int, optional  
        size of DIIS space, default 8
    netol : int, optional
        maximum difference in number of electrons at convergence,
        default 1e-6
    dtol : float, optional  
        maximum difference in subsequent density matrices at
        convergence, default 1e-6
    verbose : bool, optional
        if True, print output log, default True
    maxiter : int, optional
        maximum number of iterations of the inner Fock loop,
        default 50
    maxruns : int, optional
        maximum number of iterations of the outer Fock loop,
        default 20
    method : str, optional
        minimization algorithm to use, default 'newton'
    jac : bool, optional
        if True, use the gradient function, default True

    Returns
    -------
    se : Aux
        object containing the self-energy auxiliary space
    rdm : (n,n) ndarray
        reduced density matrix
    converged : bool
        whether the Fock loop converged
    '''

    options = _set_options(**kwargs)

    get_fock = options['fock_func']
    diis_space = options['diis_space']
    tol = options['netol']
    dtol = options['dtol']
    verbose = options['verbose']
    maxiter = options['maxiter']
    maxruns = options['maxruns']
    frozen = options['frozen']
    method = options['method']
    jac = options['jac']

    if get_fock is None:
        get_fock = lambda rdm: hf.get_fock(rdm, basis='mo')

    diis = util.DIIS(space=diis_space)
    fock = get_fock(rdm)
    nphys = se.nphys

    if not isinstance(frozen, tuple):
        frozen = (frozen, 0)
    act = slice(frozen[0], fock.shape[-1]-frozen[1])
    nelec_act = hf.nelec - frozen[0] * 2

    buf = np.zeros((nphys + se.naux,)*2, dtype=types.float64)

    log.write('----------------------------------------------------\n', verbose)
    log.write('                     Fock loop                      \n', verbose)
    log.write('  loop         RMSD  niter  nelec error      chempot\n', verbose)
    log.write('------ ------------ ------ ------------ ------------\n', verbose)


    for nrun in range(1, maxruns+1):
        se, opt = minimize(se, fock, hf.nelec, buf=buf, x0=se.chempot, tol=tol, jac=jac, method=method)

        for niter in range(1, maxiter+1):
            w, v, se.chempot, error = diag_fock_ext(se, fock, hf.nelec)

            c_occ = v[:nphys, w < se.chempot]
            rdm[act,act] = np.dot(c_occ, c_occ.T) * 2
            fock = get_fock(rdm)

            if niter > 1:
                fock = diis.update(fock)

                rmsd = np.sqrt(np.sum((rdm - rdm_prev)**2))

                if rmsd < dtol:
                    break

            rdm_prev = rdm.copy()

        log.write('%6d %12.6g %6d %12.6g %12.6f\n' % (nrun, rmsd, niter, error, se.chempot), verbose)

        if rmsd < dtol and abs(error) < tol:
            break

    converged = rmsd < dtol and abs(error) < tol

    log.write('----------------------------------------------------\n', verbose)

    return se, rdm, converged


def fock_loop_uhf(se, hf, rdm, **kwargs):
    ''' Performs the self-consistent loop of the Fock matrix in
        auxiliary GF2 for unrestricted wavefunctions.

    Parameters
    ----------
    se : tuple of (Aux, Aux)
        object containing the self-energy auxiliary space for alpha
        and beta spins
    hf : hf.UHF object
        Hartree-Fock object
    rdm : (2,n,n) ndarray
        initial reduced density matrix for alpha and beta spins
    diis_space : int, optional  
        size of DIIS space, default 8
    netol : int, optional
        maximum difference in number of electrons at convergence,
        default 1e-6
    dtol : float, optional  
        maximum difference in subsequent density matrices at
        convergence, default 1e-6
    verbose : bool, optional
        if True, print output log, default True
    maxiter : int, optional
        maximum number of iterations of the inner Fock loop,
        default 50
    maxruns : int, optional
        maximum number of iterations of the outer Fock loop,
        default 20
    method : str, optional
        minimization algorithm to use, default 'newton'
    jac : bool, optional
        if True, use the gradient function, default True

    Returns
    -------
    se : tuple of (Aux, Aux)
        object containing the self-energy auxiliary space for alpha
        and beta spins
    rdm : (2,n,n) ndarray
        reduced density matrix for alpha and beta spins
    converged : bool
        whether the Fock loop converged
    '''

    options = _set_options(**kwargs)

    get_fock = options['fock_func']
    diis_space = options['diis_space']
    tol = options['netol']
    dtol = options['dtol']
    verbose = options['verbose']
    maxiter = options['maxiter']
    maxruns = options['maxruns']
    frozen = options['frozen']
    method = options['method']
    jac = options['jac']

    if get_fock is None:
        get_fock = lambda rdm: hf.get_fock(rdm, basis='mo')

    diis = util.DIIS(space=diis_space)
    fock = get_fock(rdm)
    nphys = se[0].nphys

    if not isinstance(frozen, tuple):
        frozen = (frozen, 0)
    act = slice(frozen[0], fock.shape[-1]-frozen[1])
    nelec_act = (hf.nalph - frozen[0], hf.nbeta - frozen[0])

    bufa = np.zeros((nphys + se[0].naux,)*2, dtype=types.float64)
    bufb = np.zeros((nphys + se[1].naux,)*2, dtype=types.float64)

    log.write('-----------------------------------------------------------------\n', verbose)
    log.write('                            Fock loop                            \n', verbose)
    log.write('  loop         RMSD  niter  nelec error   chempot(a)   chempot(b)\n', verbose)
    log.write('------ ------------ ------ ------------ ------------ ------------\n', verbose)


    for nrun in range(1, maxruns+1):
        se_a, res_a = minimize(se[0], fock[0], hf.nalph, x0=se[0].chempot, tol=tol, occupancy=1, 
                               buf=bufa, method=method, jac=jac)
        se_b, res_b = minimize(se[1], fock[1], hf.nbeta, x0=se[1].chempot, tol=tol, occupancy=1, 
                               buf=bufb, method=method, jac=jac)

        for niter in range(1, maxiter+1):
            w_a, v_a, se[0].chempot, error_a = diag_fock_ext(se[0], fock[0], hf.nalph, occupancy=1)
            w_b, v_b, se[1].chempot, error_b = diag_fock_ext(se[1], fock[1], hf.nbeta, occupancy=1)
            error = (error_a, error_b)

            c_occ_a = v_a[:nphys, w_a < se[0].chempot]
            c_occ_b = v_b[:nphys, w_b < se[1].chempot]

            rdm_a = np.dot(c_occ_a, c_occ_a.T)
            rdm_b = np.dot(c_occ_b, c_occ_b.T)
            rdm[:,act,act] = np.stack((rdm_a, rdm_b))

            fock = get_fock(rdm)

            if niter > 1:
                fock = diis.update(fock)

                rmsd = np.sqrt(np.sum((rdm - rdm_prev)**2))

                if rmsd < dtol:
                    break

            rdm_prev = rdm.copy()

        log.write('%6d %12.6g %6d %12.6g %12.6f %12.6f\n' 
                  % (nrun, rmsd, niter, max(error), se[0].chempot, se[1].chempot), verbose)

        error_max = max(abs(error[0]), abs(error[1]))

        if rmsd < dtol and error_max < tol:
            break

    converged = rmsd < dtol and error_max < tol

    log.write('-----------------------------------------------------------------\n', verbose)

    return se, rdm, converged
