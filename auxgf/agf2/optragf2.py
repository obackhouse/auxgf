''' Class to perform DF-RAGF2(None,0) with efficient MPI parallel
    algorithms. Most of this code will remain self-contained for
    the sake of optimisation.
'''

import numpy as np
from pyscf import lib
from pyscf.ao2mo import _ao2mo
import functools
import ctypes

from auxgf import util, aux
from auxgf.util import types, log, mpi
#from auxgf.agf2.chempot import minimize, diag_fock_ext
from auxgf.agf2.fock import fock_loop_rhf


#TODO: save/load, damping, scs
#FIXME: should we screen auxiliaries? is it worth it?


def _set_options(options, **kwargs):
    options.update({ 'maxiter' : 50,
                     'etol' : 1e-6,
                     'wtol' : 1e-12,
                     'damping' : 0.0,
                     'delay_damping' : 0,
                     'dtol' : 1e-8,
                     'diis_space' : 8,
                     'fock_maxiter' : 50,
                     'fock_maxruns' : 20,
                     'maxblk' : 120,
                     'ss_factor' : 1.0,
                     'os_factor' : 1.0,
    })

    for key,val in kwargs.items():
        if key not in options.keys():
            raise ValueError('%s argument invalid.' % key)

    options.update(kwargs)

    options['_fock_loop'] = {
        'dtol': options['dtol'],
        'diis_space': options['diis_space'],
        'maxiter': options['fock_maxiter'],
        'maxruns': options['fock_maxruns'],
        'verbose': options['verbose'],
    }

    return options


_reshape_internal = lambda x, s1, swap, s2 : \
                           x.reshape(s1).swapaxes(*swap).reshape(s2)

_fdrv = functools.partial(_ao2mo.libao2mo.AO2MOnr_e2_drv, 
                          _ao2mo.libao2mo.AO2MOtranse2_nr_s2,
                          _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2)

to_ptr = lambda m : m.ctypes.data_as(ctypes.c_void_p)


class OptRAGF2(util.AuxMethod):
    ''' Restricted auxiliary GF2 method for (None,1) and DF integrals.

    Parameters
    ----------
    rhf : RHF
        Hartree-Fock object
    dm0 : (n,n) ndarray, optional
        initial density matrix, if None, use rhf.rdm1_mo, default
        None
    verbose : bool, optional
        if True, print output log, default True
    maxiter : int, optional
        maximum number of RAGF2 iterations, default 50
    etol : float, optional
        maximum difference in subsequent energies at convergence,
        default 1e-6
    wtol : float, optional
        minimum pole weight to be considered zero, default 1e-12
    damping : float, optional
        self-energy damping factor via
            S(i) = damping * S(i) + (1-damping) * S(i-1),
        default 0.5
    delay_damping : int, optional
        skip this number of iterations before starting to damp,
        default 0
    dtol : float, optional
        maximum difference in density matrices at convergence in the
        Fock loop, default 1e-8
    diis_space : int, optional  
        size of DIIS space for inner Fock loop, default 8
    fock_maxiter : int, optional
        maximum number of inner Fock loop iterations, default 50
    fock_maxruns : int, optional
        maximum number of outer Fock loop iterations, default 20
    maxblk : float, optional
        maximum block of density fitted integrals
    ss_factor : float, optional
        same spin factor for auxiliary build, default 1.0
    os_factor : float, optional
        opposite spin factor for auxiliary build, default 1.0

    Attributes
    ----------
    hf : RHF
        Hartree-Fock object
    nmom : tuple of int
        returns (None, 0)
    verbose : bool
        see parameters
    options : dict
        dictionary of options
    rdm1 : (n,n) ndarray
        one-particle reduced density matrix, projected into the
        physical basis
    converged : bool
        whether the method has converged
    iteration : int
        the current/final iteration reached
    se : Aux
        auxiliary representation of the self-energy
    e_1body : float
        one-body energy
    e_2body : float
        two-body energy
    e_tot : float
        total energy
    e_mp2 : float
        MP2 energy (equivalent to `e2b` at `iteration=0`)
    e_corr : float
        correlation energy i.e. `e_tot` - `hf.e_tot`

    Methods
    -------
    setup(rhf)
        constructs the object using the parameters provided to 
        `__init__` and performs the initial MP2 iteration
    get_fock(rdm1=None)
        returns the Fock matrix resulting from the current, or
        provided, density
    run()
        runs the method
    '''

    def __init__(self, rhf, **kwargs):
        super().__init__(rhf, **kwargs)

        self.options = _set_options(self.options, **kwargs)

        if mpi.mpi is None:
            log.warn('No MPI4Py installation detected, OptRAGF2 will therefore run in serial.')

        self.setup()


    @util.record_time('setup')
    def setup(self):
        super().setup()
        self.gf = self.se.new(self.hf.e, np.eye(self.hf.nao))

        if self.eri.ndim == 3:
            self.eri = lib.pack_tril(self.eri)

        log.title('Options', self.verbose)
        log.options(self.options, self.verbose)
        log.title('Input', self.verbose)
        log.molecule(self.hf.mol, self.verbose)
        log.write('Basis = %s\n' % self.hf.mol.basis, self.verbose)
        log.write('E(nuc) = %.12f\n' % self.hf.e_nuc, self.verbose)
        log.write('E(hf)  = %.12f\n' % self.hf.e_tot, self.verbose)
        log.write('nao = %d\n' % self.hf.nao, self.verbose)
        log.write('nmom = (%s, %s)\n' % self.nmom, self.verbose)

        self.run_mp2()


    @staticmethod
    def ao2mo(eri, ci, cj, sym_in='s2', sym_out='s2', maxblk=120, out=None):
        ''' Orbital transformation from density-fitted ERIs.

        Parameters
        ----------
        eri : ndarray
            Cholesky-decomposed ERI tensor
        ci : ndarray
            Coefficients for first index
        cj : ndarray
            Coefficients for second index
        sym_in : str, optional
            Input symmetry of `eri`, 's1' means no index symmetry
            and 's2' means indices are symmetric, default 's2'
        sym_out : str, optional
            Desired output symmetry, default 's2'
        out : ndarray, optional
            Output array, if None then allocate inside function,
            default None

        Returns
        -------
        out : ndarray
            Transformed Cholesky-decomposed ERI tensor
        '''

        naux = eri.shape[0]
        ijsym, nij, cij, sij = util.conc_mos(ci, cj, compact=True)
        i, j  = ci.shape[1], cj.shape[1]

        if out is None:
            if sym_out == 's2':
                out = np.zeros((naux, i*(i+1)//2), dtype=types.float64)
            else:
                out = np.zeros((naux, i*j), dtype=types.float64)

        for s in mpi.get_blocks(naux, maxblk, all_ranks=False):
            out[s] = _ao2mo.nr_e2(eri[s], cij, sij, out=out[s], aosym=sym_in, mosym=sym_out)

        out = mpi.reduce(out)

        return out


    @staticmethod
    def build_x(ixq, qja, nphys, nocc, nvir):
        ''' Builds the X array, entirely equivalent to the zeroth-
            order moment matrix of the self-energy.
        '''

        x = np.zeros((nphys, nphys), dtype=types.float64)
        buf1 = np.zeros((nphys, nocc*nvir), dtype=types.float64)
        buf2 = np.zeros((nocc*nphys, nvir), dtype=types.float64)

        for i in range(mpi.rank, nocc, mpi.size):
            xja = np.dot(ixq[i*nphys:(i+1)*nphys], qja, out=buf1)
            xia = np.dot(ixq, qja[:,i*nvir:(i+1)*nvir], out=buf2)
            xia = _reshape_internal(xia, (nocc, nphys, nvir), (0,1), (nphys, nocc*nvir))

            x = util.dgemm(xja, xja.T, alpha=2, beta=1, c=x)
            x = util.dgemm(xja, xia.T, alpha=-1, beta=1, c=x)

        x = mpi.reduce(x)

        return x


    @staticmethod
    def build_m(gf_occ, gf_vir, ixq, qja, b_inv):
        ''' Builds the M array.
        '''

        nphys = gf_occ.nphys
        nocc = gf_occ.naux
        nvir = gf_vir.naux

        m = np.zeros((nphys, nphys), dtype=types.float64)

        eo, ev = gf_occ.e, gf_vir.e
        indices = mpi.tril_indices_rows(nocc)
        pos_factor = np.sqrt(0.5)
        neg_factor = np.sqrt(1.5)

        for i in indices[mpi.rank]:
            xq = ixq[i*nphys:(i+1)*nphys]
            qa = qja[:,i*nvir:(i+1)*nvir]

            xja = np.dot(ixq[:i*nphys], qa)
            xja = _reshape_internal(xja, (i, nphys, nvir), (0,1), (nphys, i*nvir))
            xia = np.dot(xq, qja[:,:i*nvir])
            xa = np.dot(xq, qa)

            ea = eb = eo[i] + util.dirsum('i,a->ia', eo[:i], -ev).ravel()
            ec = 2 * eo[i] - ev

            va = neg_factor * (xia - xja)
            vb = pos_factor * (xia + xja)
            vc = xa

            qa = np.dot(b_inv.T, va)
            qb = np.dot(b_inv.T, vb)
            qc = np.dot(b_inv.T, vc)

            m = util.dgemm(qa * ea[None], qa.T, c=m, beta=1)
            m = util.dgemm(qb * eb[None], qb.T, c=m, beta=1)
            m = util.dgemm(qc * ec[None], qc.T, c=m, beta=1)

        m = mpi.reduce(m)

        return m


    @staticmethod
    def build_part(gf_occ, gf_vir, eri, sym_in='s2'):
        ''' Builds the truncated occupied (or virtual) self-energy.

        Parameters
        ----------
        gf_occ : Aux
            Occupied (or virtual) Green's function
        gf_vir : Aux
            Virtual (or occupied) Green's function
        eri : ndarray
            Cholesky-decomposed DF ERI tensor
        sym_in : str, optional
            Symmetry of `eri`, default 's2'

        Returns
        -------
        se : Aux
            Occupied (or virtual) truncated self-energy
        '''

        syms = dict(sym_in=sym_in, sym_out='s1')
        nphys = gf_occ.nphys
        nocc = gf_occ.naux
        nvir = gf_vir.naux

        ixq = OptRAGF2.ao2mo(eri, gf_occ.v, np.eye(nphys), **syms).T
        qja = OptRAGF2.ao2mo(eri, gf_occ.v, gf_vir.v, **syms)

        x = OptRAGF2.build_x(ixq, qja, nphys, nocc, nvir)
        b = np.linalg.cholesky(x).T
        b_inv = np.linalg.inv(b)
        m = OptRAGF2.build_m(gf_occ, gf_vir, ixq, qja, b_inv)

        e, c = util.eigh(m)
        c = np.dot(b.T, c[:nphys])

        se = gf_occ.new(e, c)

        return se


    @util.record_time('build')
    def build(self):
        e, c = self.se.eig(self.get_fock())
        self.gf = self.se.new(e, c[:self.nphys])

        gf_occ = self.gf.as_occupied()
        gf_vir = self.gf.as_virtual()

        se_occ = self.build_part(gf_occ, gf_vir, self.eri)
        se_vir = self.build_part(gf_vir, gf_occ, self.eri)

        self.se = se_occ + se_vir


    @util.record_time('fock')
    def fock_loop(self):
        fock_opts = self.options['_fock_loop']

        se, rdm1, converged = fock_loop_rhf(self.se, self.hf, self.rdm1, **fock_opts)

        w, v = self.se.eig(self.get_fock())
        self.gf = self.se.new(w, v[:self.nphys])

        if converged:
            log.write('Fock loop converged.\n', self.verbose)
            log.write('Chemical potential = %.6f\n' % self.chempot, self.verbose)
        else:
            log.write('Fock loop did not converge.\n', self.verbose)
            log.write('Chemical potential = %.6f\n' % self.chempot, self.verbose)


    @util.record_time('energy')
    @util.record_energy('mp2')
    def energy_mp2(self):
        emp2 = aux.energy.energy_mp2_aux(self.hf.e, self.se)

        log.write('E(mp2) = %.12f\n' % emp2, self.verbose)

        return emp2

    @util.record_time('energy')
    @util.record_energy('1b')
    def energy_1body(self):
        e1b = self.hf.energy_1body(self.h1e, self.rdm1, self.get_fock())
        e1b += self.hf.mol.e_nuc

        log.write('E(1b)  = %.12f\n' % e1b, self.verbose)

        return e1b

    @util.record_time('energy')
    @util.record_energy('2b')
    def energy_2body(self):
        e_qmo, v_qmo = self.se.eig(self.get_fock())
        self.gf = self.se.new(e_qmo, v_qmo[:self.nphys])

        e2b = 0.0

        for l in range(mpi.rank, self.gf.nocc, mpi.size):
            vxl = self.gf.v[:,l]
            vxk = self.se.v[:,self.se.nocc:]

            dlk = 1.0 / (self.gf.e[l] - self.se.e[self.se.nocc:])

            e2b += util.einsum('xk,yk,x,y,k->', vxk, vxk, vxl, vxl, dlk)

        e2b = 2.0 * np.ravel(e2b)[0]
        e2b = mpi.reduce(e2b)

        log.write('E(2b)  = %.12f\n' % e2b, self.verbose)

        return e2b

    @util.record_time('energy')
    @util.record_energy('tot')
    def energy_tot(self):
        etot = self.e_1body + self.e_2body

        log.write('E(tot) = %.12f\n' % etot, self.verbose)

        return etot

    def energy(self):
        self.energy_1body()
        self.energy_2body()
        self.energy_tot()


    def get_fock(self, rdm1=None):
        ''' Returns the Fock matrix resulting from the current, or
            provided, density.
        '''

        if rdm1 is None:
            rdm1 = self.rdm1

        nphys = self.nphys
        naux = self.eri.shape[0]
        maxblk = self.options['maxblk']

        j = np.zeros((nphys*(nphys+1)//2), dtype=types.float64)
        k = np.zeros((nphys, nphys), dtype=types.float64)

        rdm1_tril = lib.pack_tril(rdm1 + np.tril(rdm1, k=-1))
        c_int = ctypes.c_int
        args = (c_int(nphys), (c_int*4)(0, nphys, 0, nphys), lib.c_null_ptr(), c_int(0))
        buf = np.empty((2, maxblk, nphys, nphys))

        for s in mpi.get_blocks(naux, maxblk, all_ranks=False):
            eri1 = self.eri[s]
            naux_block = eri1.shape[0]

            rho = np.dot(eri1, rdm1_tril)
            j += np.dot(rho, eri1)

            buf1 = buf[0,:naux_block]
            _fdrv(to_ptr(buf1), to_ptr(eri1), to_ptr(rdm1), c_int(naux_block), *args)

            buf2 = lib.unpack_tril(eri1, out=buf[1])
            k = util.dgemm(buf1.reshape((-1, nphys)).T, buf2.reshape((-1, nphys)), c=k, beta=1)

        j = mpi.reduce(j)
        k = mpi.reduce(k)

        j = lib.unpack_tril(j).reshape(rdm1.shape)
        k = k.reshape(rdm1.shape)

        f = self.h1e + j - 0.5 * k

        return f


    def run_mp2(self):
        log.iteration(0, self.verbose)

        self.build()
        self.energy_mp2()


    def run(self):
        maxiter = self.options['maxiter']
        etol = self.options['etol']

        for self.iteration in range(1, maxiter+1):
            log.iteration(self.iteration, self.verbose)

            self.fock_loop()
            self.build()
            self.energy()

            if self.iteration > 1:
                e_dif = abs(self._energies['tot'][-2] - self._energies['tot'][-1])

                if e_dif < etol and self.converged:
                    break

                self.converged = e_dif < etol

        if self.converged:
            log.write('\nAuxiliary GF2 converged after %d iterations.\n' 
                      % self.iteration, self.verbose)
        else:
            log.write('\nAuxiliary GF2 failed to converge.\n', self.verbose)

        self._timings['total'] = self._timings.get('total', 0.0) + self._timer.total()
        log.title('Timings', self.verbose)
        log.timings(self._timings, self.verbose)

        return self


    @property
    def e_2body(self):
        if self.iteration:
            return self._energies['2b'][-1]
        else:
            return self.e_mp2

    @property
    def e_tot(self):
        if self.iteration:
            return self._energies['tot'][-1]
        else:
            return self.e_hf + self.e_mp2

    @property
    def e_mp2(self):
        return self._energies['mp2'][-1]


    @property
    def nmom(self):
        return (None, 0)
