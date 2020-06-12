''' Functions to perform the Fock loop used in auxiliary GF2.
'''

import numpy as np
from scipy.optimize import minimize_scalar

from auxgf import util
from auxgf.util import types, log, mpi

# If anyone is reading this I sincerely apologise for how shit this code is


def _set_options(**kwargs):
    options = { 'diis_space': 8,
                'netol': 1e-6,
                'dtol': 1e-6,
                'verbose': True,
                'maxiter': 50,
                'maxruns': 20,
                'frozen': 0,
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

    diis = util.DIIS(space=options['diis_space'])

    fock = hf.get_fock(rdm, basis='mo')

    e0 = se.e.copy()
    v0 = se.v.copy()
    chempot = se.chempot
    nphys = se.nphys

    frozen = options['frozen']
    if not isinstance(frozen, tuple):
        frozen = (frozen, 0)
    act = slice(frozen[0], fock.shape[-1]-frozen[1])
    nelec_act = hf.nelec - frozen[0] * 2


    def _diag_fock_ext(chempot):
        _fock_ext = _get_fock_ext(fock[act,act], e0, v0, chempot)
        _w, _v = util.eigh(_fock_ext)
        _chempot, _error = util.find_chempot(nphys, nelec_act, h=(_w, _v))
        return _w, _v, _chempot, _error

    def _obj(x):
        return abs((_diag_fock_ext(x))[-1])

    def _minimize(homo, lumo):
        return minimize_scalar(_obj, bounds=(homo, lumo), method='bounded',
                        options={ 'maxiter': 1000, 'xatol': options['netol']})


    log.write('%52s\n' % ('-'*52), options['verbose'])
    log.write('Fock loop'.center(52) + '\n', options['verbose'])
    log.write('%6s %12s %6s %12s %12s\n' % ('loop', 'RMSD', 'niter',
              'nelec_error', 'chempot'), options['verbose'])
    log.write('%6s %12s %6s %12s %12s\n' % ('-'*6, '-'*12, '-'*6, '-'*12, 
              '-'*12), options['verbose'])


    for nrun in range(1, options['maxruns']+1):
        w, v, chempot, error = _diag_fock_ext(0.0)

        homo = util.amax(w[w < chempot])
        lumo = util.amin(w[w >= chempot])

        if not (homo is np.nan or lumo is np.nan):
            res = _minimize(homo, lumo)
            e0 -= res.x

        for niter in range(1, options['maxiter']+1):
            w, v, chempot, error = _diag_fock_ext(0.0)

            c_occ = v[:nphys, w < chempot]
            rdm[act,act] = np.dot(c_occ, c_occ.T) * 2
            fock = hf.get_fock(rdm, basis='mo')

            if niter > 1:
                fock = diis.update(fock)

                rmsd = np.sqrt(np.sum((rdm - rdm_prev)**2))

                if rmsd < options['dtol']:
                    break

            rdm_prev = rdm.copy()

        log.write('%6d %12.6g %6d %12.6g %12.6f\n' % 
                  (nrun, rmsd, niter, error, chempot), options['verbose'])

        if rmsd < options['dtol'] and abs(error) < options['netol']:
            break

    # TODO these are sometimes flagged but doesn't seem to be a problem...
    #if homo is np.nan:
    #    util.log.warn('Could not find a HOMO in Fock loop.')
    #elif lumo is np.nan:
    #    util.log.warn('Could not find a LUMO in Fock loop.')

    converged = rmsd < options['dtol'] and abs(error) < options['netol']

    log.write('%52s\n' % ('-'*52), options['verbose'])

    se = se.new(e0, v0, chempot=chempot)

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

    diis = util.DIIS(space=options['diis_space'])

    fock = hf.get_fock(rdm, basis='mo')

    e0 = (se[0].e.copy(), se[1].e.copy())
    v0 = (se[0].v.copy(), se[1].v.copy())
    chempot = (se[0].chempot, se[1].chempot)
    nphys = se[0].nphys

    frozen = options['frozen']
    if not isinstance(frozen, tuple):
        frozen = (frozen, 0)
    act = slice(frozen[0], fock.shape[-1]-frozen[1])
    nelec_act = (hf.nalph - frozen[0], hf.nbeta - frozen[0])


    def _diag_fock_ext(chempot, ab):
        _fock_ext = _get_fock_ext(fock[ab][act,act], e0[ab], v0[ab], chempot)
        _w, _v = util.eigh(_fock_ext)
        _chempot, _error = util.find_chempot(nphys, nelec_act[ab], h=(_w, _v),
                                             occupancy=1.0)
        return _w, _v, _chempot, _error

    def _obj(x, ab):
        return abs((_diag_fock_ext(x, ab))[-1])

    def _minimize(homo, lumo, ab):
        return minimize_scalar(_obj, bounds=(homo, lumo), method='bounded',
               options={ 'maxiter': 1000, 'xatol': options['netol']}, args=(ab))


    log.write('%65s\n' % ('-'*65), options['verbose'])
    log.write('Fock loop'.center(65) + '\n', options['verbose'])
    log.write('%6s %12s %6s %12s %12s %12s\n' % ('loop', 'RMSD', 'niter',
              'nelec error', 'chempot(a)', 'chempot(b)'), options['verbose'])
    log.write('%6s %12s %6s %12s %12s %12s\n' % ('-'*6, '-'*12, '-'*6, '-'*12,
              '-'*12, '-'*12), options['verbose'])


    for nrun in range(1, options['maxruns']+1):
        w_a, v_a, chempot_a, error_a = _diag_fock_ext(0.0, 0)
        w_b, v_b, chempot_b, error_b = _diag_fock_ext(0.0, 1)

        chempot = (chempot_a, chempot_b)
        error = (error_a, error_b)

        homo_a = util.amax(w_a[w_a < chempot[0]])
        lumo_a = util.amin(w_b[w_b >= chempot[0]])
        homo_b = util.amax(w_a[w_a < chempot[1]])
        lumo_b = util.amin(w_b[w_b >= chempot[1]])

        if not (homo_a is np.nan or homo_b is np.nan 
                or lumo_a is np.nan or lumo_b is np.nan):
            res_a = _minimize(homo_a, lumo_a, 0)
            res_b = _minimize(homo_b, lumo_b, 1)
            e0 = (e0[0] - res_a.x, e0[1] - res_b.x)

        for niter in range(1, options['maxiter']+1):
            w_a, v_a, chempot_a, error_a = _diag_fock_ext(0.0, 0)
            w_b, v_b, chempot_b, error_b = _diag_fock_ext(0.0, 1)

            chempot = (chempot_a, chempot_b)
            error = (error_a, error_b)

            c_occ_a = v_a[:nphys, w_a < chempot[0]]
            c_occ_b = v_b[:nphys, w_b < chempot[1]]

            rdm_a = np.dot(c_occ_a, c_occ_a.T)
            rdm_b = np.dot(c_occ_b, c_occ_b.T)
            rdm[:,act,act] = np.stack((rdm_a, rdm_b))

            fock = hf.get_fock(rdm, basis='mo')

            if niter > 1:
                fock = diis.update(fock)

                rmsd = np.sqrt(np.sum((rdm - rdm_prev)**2))

                if rmsd < options['dtol']:
                    break

            rdm_prev = rdm.copy()

        log.write('%6d %12.6g %6d %12.6g %12.6f %12.6f\n' % 
                  (nrun, rmsd, niter, max(error), *chempot), options['verbose'])

        error_max = max(abs(error[0]), abs(error[1]))

        if rmsd < options['dtol'] and error_max < options['netol']:
            break

    #if homo_a is np.nan or homo_b is np.nan:
    #    util.log.warn('Could not find a HOMO in Fock loop.')
    #elif lumo_a is np.nan or lumo_b is np.nan:
    #    util.log.warn('Could not find a LUMO in Fock loop.')

    converged = rmsd < options['dtol'] and error_max < options['netol']

    log.write('%65s\n' % ('-'*65), options['verbose'])

    se = (se[0].new(e0[0], v0[0], chempot=chempot[0]), 
          se[1].new(e0[1], v0[1], chempot=chempot[1]))

    return se, rdm, converged
    

if __name__ == '__main__':
    pass
