''' Main class to control auxiliary parameters.
'''

import numpy as np
import copy
import pickle

from auxgf import util, grids
from auxgf.util import types
from auxgf.aux import merge, setrunc, gftrunc, fit, eig


_get_v_dtype = lambda v: types.complex128 if np.iscomplexobj(v) else types.float64


class Aux:
    ''' Class to contain a set of fully-causal auxiliary poles. These
        define a spectrum which consists of a frequency-dependent set
        of matrices in the physical space.

    Parameters
    ----------
    e : (n) ndarray
        poles of the spectrum, i.e. auxiliary energies
    v : (m,n) ndarray
        couplings of the poles to a physical space
    chempot : float, optional
        chemical potential on the physical space
        (default 0.0)

    Attributes
    ----------
    e : (n) ndarray
        poles of the spectrum, i.e. auxiliary energies
    v : (m,n) ndarray
        couplings of the poles to a physical space
    e_occ : (n) ndarray
        occupied auxiliary energies
    v_occ : (m,n) ndarray
        occupied couplings 
    e_vir : (n) ndarray
        virutal auxiliary energies
    v_vir : (m,n) ndarray
        virtual couplings
    chempot : float
        chemical potential on the physical space
    nphys : int
        number of physical degrees of freedom
    naux : int
        number of auxiliaries
    nocc : int
        number of occupied auxiliaries
    nvir : int
        number of virtual auxiliaries

    Methods
    -------
    build_denominator(grid, e, v, chempot=0.0, ordering='feynman')
        static method to build the denominator required by the
        spectral function
    build_spectrum(grid, e, v, chempot=0.0, ordering='feynman')
        static method to build a spectral function
    build_denominator(grid, e, v, chempot=0.0, ordering='feynman')
        static method to build a derivative of the spectral function
        with respect to frequency
    as_spectrum(grid, ordering='feynman')
        express the spectral function as a frequency-dependent matrix
    as_derivative(grid, ordering='feynman')
        express the derivative of a spectral function as a frequency-
        dependent matrix
    as_hamiltonian(h_phys, chempot=0.0)
        expresses the auxiliaries as a matrix in the 'extened Fock'
        formalism
    as_occupied()
        returns a new object with only the occupied auxiliaries
    as_virtual()
        returns a new object with only the virtual auxiliaries
    as_window(e_min, e_max)
        returns a new object with only the auxiliaries between a given
        energy window [e_min, e_max)
    dot(h_phys, vec)
        dot-product of `as_hamiltonian` form with vector `vec` in
        quadratic time
    eig(h_phys, chempot=0.0)
        diagonalises `as_hamiltonian`
    moment(n)
        builds the nth moment of the spectral distribution
    merge(etol=1e-10, wtol=1e-12)
        in-principle exact reduction of the auxiliaries removing
        linear dependencies and poles with no weight on the physical
        system
    se_compress(h_phys, nmom)
        compresses the auxiliaries via moments of the self-energy
    gf_compress(h_phys, nmom)
        compresses the auxiliaries via moments of the Green's function
    compress(h_phys, nmom
        compresses the auxiliaries via the hybrid algorithm
    fit(target, grid, hessian=True, opts={}, test_grad=False, test_hess=False)
        runs the auxiliary fitting procedure
    sort(which='e')
        sorts the auxiliaries by the energies, weights, or both
    copy()
        returns a copy of the object
    save(filename)
        saves the object the the disk
    load(filename)
        class method to load the object from the disk
    memsize()
        approximate size of the object
    new(e, v)
        returns a new object with different poles, inheriting other
        attributes
    '''


    def __init__(self, e, v, chempot=0.0):
        self._setup(e, v, chempot=chempot)


    def _setup(self, e, v, chempot=0.0):
        self._ener = np.array(e, order='C', copy=False, dtype=types.float64)
        self._coup = np.array(v, order='C', copy=False, dtype=_get_v_dtype(v))

        self.chempot = 0.0 if None else chempot


    @staticmethod
    def build_denominator(grid, e, v, chempot=0.0, ordering='feynman'):
        ''' Builds the denominator required to build a spectrum on a
            frequency grid suing the auxiliaries.

        Parameters
        ----------
        grid : (n) ImFqGrid, ImFqQuad or ReFqGrid
            grid object
        e : (m) ndarray
            auxiliary energies
        v : (k,m) ndarray
            auxiliary couplings
        chempot : float, optional
            chemical potential
        ordering : str
            ordering of the poles {'feynman', 'advanced', 'retarded'}
            (default 'feynman')

        Returns
        -------
        sf : (n,m,m) ndarray
            spectrum expressed on `grid`
        '''

        e_p = e - chempot
        w = grid.prefac * grid.values

        if ordering == 'feynman':
            s = np.sign(e_p)
        elif ordering == 'advanced':
            s = np.ones(e_p.shape, dtype=types.int64)
        elif ordering == 'retarded':
            s = -np.ones(e_p.shape, dtype=types.int64)
        else:
            raise ValueError

        denom = util.outer_sum([w, -e_p + s * grid.eta * 1.0j])
        denom = 1.0 / denom

        return denom


    @staticmethod
    def build_spectrum(grid, e, v, chempot=0.0, ordering='feynman', blksize=100):
        ''' Builds a spectrum on a frequency grid using the auxiliaries.

        Parameters
        ----------
        grid : (n) ImFqGrid, ImFqQuad or ReFqGrid
            grid object
        e : (m) ndarray
            auxiliary energies
        v : (k,m) ndarray
            auxiliary couplings
        chempot : float, optional
            chemical potential
        ordering : str, optional
            ordering of the poles {'feynman', 'advanced', 'retarded'},
            default 'feynman'
        blksize : int, optional
            number of frequency points per block, default 100

        Returns
        -------
        sf : (n,m,m) ndarray
            spectrum expressed on `grid`
        '''

        #TODO write in C, memory-efficient implementation in Python
        # may need too much python for loops to be fast

        es = e - chempot
        freq = grid.prefac * grid.values
        nphys = v.shape[0]

        if ordering == 'feynman':
            s = np.sign(es)
        elif ordering == 'advanced':
            s = np.ones(es.shape, dtype=types.int64)
        elif ordering == 'retarded':
            s = -np.ones(es.shape, dtype=types.int64)
        else:
            raise ValueError

        es = es - s * grid.eta * 1.0j

        sf = np.zeros((grid.size, nphys, nphys), dtype=types.complex128)

        p1 = 0
        for block in range(0, len(freq), blksize):
            p0, p1 = p1, p1 + blksize
            denom = 1.0 / util.outer_sum([freq[p0:p1], -es])
            sf[p0:p1] += util.einsum('xk,yk,wk->wxy', v, v, denom)

        return sf


    @staticmethod
    def build_derivative(grid, e, v, chempot=0.0, ordering='feynman', blksize=100):
        ''' Builds the derivative of the spectrum with respect to
            frequency on a frequency grid using the auxiliaries.

        Parameters
        ----------
        grid : (n) ImFqGrid, ImFqQuad or ReFqGrid
            grid object
        e : (m) ndarray
            auxiliary energies
        v : (k,m) ndarray
            auxiliary couplings
        chempot : float, optional
            chemical potential
        ordering : str
            ordering of the poles {'feynman', 'advanced', 'retarded'},
            default 'feynman'
        blksize : int, optional
            number of frequency points per block, default 100

        Returns
        -------
        dsf : (n,m,m) ndarray
            derivative of spectrum expressed on `grid`
        '''

        es = e - chempot
        freq = grid.prefac * grid.values
        nphys = v.shape[0]

        if ordering == 'feynman':
            s = np.sign(es)
        elif ordering == 'advanced':
            s = np.ones(es.shape, dtype=types.int64)
        elif ordering == 'retarded':
            s = -np.ones(es.shape, dtype=types.int64)
        else:
            raise ValueError

        es = es - s * grid.eta * 1.0j

        dsf = np.zeros((grid.size, nphys, nphys), dtype=types.complex128)

        p1 = 0
        for block in range(0, len(freq), blksize):
            p0, p1 = p1, p1 + blksize
            denom = 1.0 / util.outer_sum([freq[p0:p1], -es]) ** 2
            dsf[p0:p1] -= util.einsum('xk,yk,wk->wxy', v, v, denom)

        return dsf


    def as_spectrum(self, grid, ordering='feynman'):
        ''' Expresses the poles as the spectrum on a frequency grid.

        Parameters
        ----------
        grid : (n) ImFqGrid, ImFqQuad or ReFqGrid
            grid object
        ordering : str
            ordering of the poles {'feynman', 'advanced', 'retarded'}
            (default 'feynman')

        Returns
        -------
        sf : (n,m,m) ndarray
            spectrum expressed on `grid`
        '''

        return self.build_spectrum(grid, self.e, self.v, chempot=self.chempot, ordering=ordering)


    def as_derivative(self, grid, ordering='feynman'):
        ''' Expresses the poles as a derivative of the spectrum with
            respect to frequency on a frequency grid.

        Parameters
        ----------
        grid : (n) ImFqGrid, ImFqQuad or ReFqGrid
            grid object
        ordering : str
            ordering of the poles {'feynman', 'advanced', 'retarded'}
            (default 'feynman')

        Returns
        -------
        dsf : (n,m,m) ndarray
            derivative of spectrum expressed on `grid`
        '''

        return self.build_derivative(grid, self.e, self.v, chempot=self.chempot, ordering=ordering)


    def as_hamiltonian(self, h_phys, chempot=0.0, out=None):
        ''' Expresses the auxiliaries as an extended Hamiltonian.

        Parameters
        ----------
        h_phys : (n,n) ndarray
            physical Hamiltonian
        chempot : float, optional
            chemical potential on the auxiliary space
        out : (m,m) ndarray, optional
            array to store output for efficient control of memory

        Returns
        -------
        h_ext : (m,m) ndarray
            extended Hamiltonian

        Raises
        ------
        ValueError
            when the physical space of `h_phys` and the couplings do
            not share dimension size
        '''

        if h_phys.shape != (self.nphys, self.nphys):
            raise ValueError('physical space of h_phys and couplings must match.')


        if out is None:
            e = self.e - chempot
            out = np.block([[h_phys, self.v], [self.v.conj().T, np.diag(e)]])
        else:
            sp = slice(None, self.nphys)
            sa = slice(self.nphys, None)

            out[sp,sp] = h_phys
            out[sp,sa] = self.v
            out[sa,sp] = self.v.conj().T
            out[sa,sa][np.diag_indices(self.naux)] = self.e - chempot

        return out


    def as_occupied(self):
        ''' Returns a new Aux object with only the occupied auxiliaries.

        Returns
        -------
        occ : Aux
            occupied auxiliaries
        '''

        occ = self.copy()

        mask = self.e < self.chempot

        occ._ener = np.array(occ.e[mask], order='C', copy=False, dtype=types.float64)
        occ._coup = np.array(occ.v[:,mask], order='C', copy=False, dtype=self.v.dtype)

        return occ


    def as_virtual(self):
        ''' Returns a new Aux object with only the virtual auxiliaries.

        Returns
        -------
        vir : Aux
            virtual auxiliaries
        '''

        vir = self.copy()
        
        mask = self.e >= self.chempot

        vir._ener = np.array(vir.e[mask], order='C', copy=False, dtype=types.float64)
        vir._coup = np.array(vir.v[:,mask], order='C', copy=False, dtype=self.v.dtype)

        return vir


    def as_window(self, e_min, e_max):
        ''' Returns a new Aux object with only the auxiliaries between
            an energy window [e_min, e_max).

        Returns
        -------
        aux : Aux
            window auxiliaries
        '''

        mask = np.logical_and(self.e >= e_min, self.e < e_max)

        aux = self.copy()

        aux._ener = np.array(aux.e[mask], order='C', copy=False, dtype=types.float64)
        aux._coup = np.array(aux.v[:,mask], order='C', copy=False, dtype=self.v.dtype)

        return aux


    def dot(self, h_phys, vec, chempot=0.0):
        ''' Dot-product of `self.as_hamiltonian(h_phys)` with `vec`,
            without having to explicitly construct the extended
            Hamiltonian.

        Parameters
        ----------
        h_phys : (n,n) ndarray
            physical Hamiltonian
        vec : (m,...) ndarray
            vector
        chempot : float, optional
            chemical potential on the auxiliary space

        Returns
        -------
        out : (m,...) ndarray
            result of dot-product

        Raises
        ------
        ValueError
            when the physical space of `h_phys` and the couplings do
            not share dimension size
        '''

        if h_phys.shape != (self.nphys, self.nphys):
            raise ValueError('physical space of h_phys and couplings must match.')

        dtype = np.result_type(self.v.dtype, vec.dtype)

        input_shape = vec.shape
        vec = vec.reshape((self.naux + self.nphys, -1))

        sp = slice(None, self.nphys)
        sa = slice(self.nphys, None)

        out = np.zeros((vec.shape), dtype=dtype)

        out[sp]  = np.dot(h_phys, vec[sp])
        out[sp] += np.dot(self.v, vec[sa])
        
        out[sa]  = np.dot(vec[sp].T, self.v).conj().T  #TODO: no .conj() on vec[sp].T right?
        out[sa] += (self.e[:,None] - chempot) * vec[sa]

        return out.reshape(input_shape)


    def eig(self, h_phys, **kwargs):
        ''' Diagonalises `self.as_hamiltonian(h_phys)`.

        Parameters
        ----------
        h_phys : (n,n) ndarray
            physical Hamiltonian
        chempot : float, optional
            chemical potential on the auxiliary space
        nroots : int, optional
            number of eigenvalues required, default -1 (returns all)
        which : str, optional
            which eigenvalues to compute (see SciPy), default 'SM'
        tol : float, optional
            convergence tolerance, default 1e-14
        maxiter : int, optional
            maximum number of iterations, default 10*dim
        ntrial : int, optional
            maximum number of trial vectors, default 
            min(dim, max(2*nroots+1, 20))

        Returns
        -------
        w : (m) ndarray
            eigenvalues
        v : (m,m) ndarray
            eigenvectors

        Raises
        ------
        ValueError
            when the physical space of `h_phys` and the couplings do
            not share dimension size
        '''

        if h_phys.shape != (self.nphys, self.nphys):
            raise ValueError('physical space of h_phys and couplings must match.')

        w, v = eig.eigh(self, h_phys, **kwargs)

        return w, v


    def moment(self, n):
        ''' Builds the nth moment of the spectral distribution.

        Parameters
        ----------
        n : int or (m) array
            single moment or an array of moment orders

        Returns
        -------
        moms : (k,k) ndarray or (m,k,k) ndarray
            array containing nth moments
        '''

        n = np.asarray(n, dtype=types.int64)
        n = n.reshape(n.size)

        en = self.e[None] ** n[:,None]
        moms = util.einsum('xk,yk,nk->nxy', self.v, self.v, en)

        return np.squeeze(moms)


    def merge(self, etol=1e-10, wtol=1e-12):
        raise NotImplementedError #FIXME
        ''' Performs an in-principle exact reduction of the auxiliaries
            which have linear dependencies or negligible weight.

        Parameters
        ----------
        etol : float, optional
            maximum difference in degenerate energies (default 1e-10)
        wtol : float, optional
            maximum weight to be considered negligible (default 1e-12)

        Returns
        -------
        red : Aux
            reduced auxiliaries
        '''

        return merge.aux_merge_exact(self, etol=etol, wtol=wtol)


    def se_compress(self, h_phys, nmom, run_anyway=False, qr='cholesky'):
        ''' Compresses the auxiliaries via the associated self-energy.
            Compression is performed out-of-place.

        Parameters
        ----------
        h_phys : (n,n) ndarray
            physical space Hamiltonian
        nmom : int
            number of moments
        run_anyway : bool, optional
            if number of resulting auxiliaries will be more than the
            current number, run the function anyway, default False
        qr : str, optional
            type of QR solver to use for SE truncation {'cholesky', 
            'numpy', 'scipy', 'unsafe'}, default 'cholesky'

        Returns
        -------
        red : Aux
            reduced auxiliaries
        '''

        if not run_anyway:
            if 2*self.nphys*(nmom+1) > self.naux:
                return self.copy()

        red = setrunc.run(self, h_phys, nmom, qr=qr)

        return red


    def gf_compress(self, h_phys, nmom, method='power', beta=100, chempot=0.0, run_anyway=False):
        ''' Compresses the auxiliaries via the associated Green's 
            function. Compression is performed out-of-place.

        Parameters
        ----------
        h_phys : (n,n) ndarray
            physical space Hamiltonian
        nmom : int
            number of moments
        method : str, optional
            kernel method {'power', 'legendre'}, default 'power'
        beta : float, optional
            inverse temperature, required for `method='legendre'`
        chempot : float, optional
            chemical potential, required for `method='legendre'`
        run_anyway : bool, optional
            if number of resulting auxiliaries will be more than the
            current number, run the function anyway, default False

        Returns
        -------
        red : Aux
            reduced auxiliaries
        '''

        if not run_anyway:
            if self.nphys*(2*nmom+1) > self.naux:
                return self.copy()

        red = gftrunc.run(self, h_phys, nmom, method=method, beta=beta, chempot=chempot)

        return red


    def compress(self, h_phys, nmom, method='power', beta=100, chempot=0.0, qr='cholesky'):
        ''' Compresses the auxiliaries via the hybird algorithm.
            Compression is performed out-of-place.

        Parameters
        ----------
        h_phys : (n,n) ndarray
            physical space Hamiltonian
        nmom : tuple of int or None
            number of moments, where first element is the number of
            Green's function moments, and second number if the number 
            of initial self-energy moments (either can be None)
        method : str, optional
            GF kernel method {'power', 'legendre'}, default 'power'
        beta : float, optional
            GF inverse temperature, required for `method='legendre'`
        chempot : float, optional
            chemical potential, required for `method='legendre'`
        qr : str, optional
            type of QR solver to use for SE truncation {'cholesky', 
            'numpy', 'scipy', 'unsafe'}, default 'cholesky'

        Returns
        -------
        red : Aux
            reduced auxiliaries
        '''

        gf_opts = {
            'method': method,
            'beta': beta,
            'chempot': self.chempot,
        }

        if nmom is None or nmom == (None, None):
            return self.copy()
        elif nmom[0] == None:
            return self.se_compress(h_phys, nmom[1], qr=qr)
        elif nmom[1] == None:
            return self.gf_compress(h_phys, nmom[0], **gf_opts)

        # I don't want to flag for intermediate self.merge call, if
        # this is desired then the user should separately call
        # self.se_compress and self.gf_compress

        red = self.se_compress(h_phys, nmom[1], qr=qr)
        red = red.gf_compress(h_phys, nmom[0], **gf_opts)

        return red


    def fit(self, target, grid, **kwargs): # pragma: no cover
        raise NotImplementedError #FIXME
        ''' Runs the auxiliary fitting procedure. 

        Parameters
        ----------
        see auxgf.aux.fit.run

        Returns
        -------
        aux_fit : Aux
            fitted auxiliaries
        '''

        aux_fit, res = fit.run(self, target, grid, **kwargs)

        return aux_fit


    def sort(self, which='e'):
        ''' Sorts the auxiliaries by their energies, weights or both.

        Parameters
        ----------
        which : str
            which value to sort by {'e', 'w', 'e,w', 'w,e'} 
            (default 'e')
        '''

        if which == 'e':
            mask = self.e.argsort()
        elif which == 'w':
            mask = self.w.argsort()
        elif which == 'e,w':
            mask = np.lexsort((self.w, self.e))
        elif which == 'w,e':
            mask = np.lexsort((self.e, self.w))

        self._ener = np.array(self._ener[mask], order='C', copy=False, dtype=types.float64)
        self._coup = np.array(self._coup[:,mask], order='C', copy=False, dtype=self.v.dtype)


    def copy(self):
        ''' Returns a copy of the Aux object.

        Returns
        -------
        aux : Aux
            copy of the auxiliaries
        '''

        aux = Aux(self.e.copy(), self.v.copy())

        for key,val in self.__dict__.items():
            if key[0] != '_':
                setattr(aux, key, copy.deepcopy(val))

        return aux

    __copy__ = copy


    def save(self, filename):
        ''' Saves the object using pickle.
        '''

        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)


    @classmethod
    def load(cls, filename):
        ''' Loads the object using pickle.

        Returns
        -------
        aux : Aux
            loaded auxiliaries
        '''

        with open(filename, 'rb') as f:
            dic = pickle.load(f)

        aux = cls(None, None)

        for item in dic.items():
            setattr(aux, *item)

        return aux


    def memsize(self):
        ''' Returns an approximate size of the object.

        Returns
        -------
        size : float
            size in GB
        '''

        b = self.e.nbytes + self.v.nbytes + 8
        gb = b / 1e9

        return gb

    __sizeof__ = memsize


    def new(self, e, v, chempot=None):
        ''' Returns a new Aux object with different energies and
            couplings, inheriting all other attributes.

        Parameters
        ----------
        e : (n) array
            new energies
        v : (m,n) array
            new couplings
        chempot : float, optional
            used as chemical potential instead of inheriting

        Returns
        -------
        aux : Aux
            new auxiliaries
        '''

        if chempot is None:
            chempot = self.chempot

        e = np.array(e, order='C', copy=False, dtype=types.float64)
        v = np.array(v, order='C', copy=False, dtype=_get_v_dtype(v))

        aux = Aux(e, v, chempot=chempot)

        return aux


    def __add__(self, other):
        ''' Combines two Aux objects. All non-combined attributes
            (i.e. chempot) are inherited from the left-hand operand.

        Parameters
        ----------
        other : Aux
            right-hand operand

        Returns
        -------
        aux : Aux
            combined auxiliaries

        Raises
        ------
        ValueError
            if the two auxiliaries have a different physical space
            dimension
        '''

        if self.nphys != other.nphys:
            raise ValueError('Cannot combine auxiliaries with different physical space dimensions.')

        e = np.concatenate((self.e, other.e), axis=0)
        v = np.concatenate((self.v, other.v), axis=1)

        aux = Aux(None, None)

        for item in self.__dict__.items():
            setattr(aux, *item)

        aux._ener = np.array(e, order='C', copy=False, dtype=types.float64)
        aux._coup = np.array(v, order='C', copy=False, dtype=_get_v_dtype(v))

        return aux


    def __eq__(self, other):
        ''' Checks if the Aux objects represent the same spectrum.

        Parameters
        ----------
        other : Aux
            right-hand operand

        Returns
        -------
        check : bool
            whether they are equal
        '''

        grid = grids.ImFqGrid(32, beta=8)

        sf1 = self.as_spectrum(grid)
        sf2 = other.as_spectrum(grid)

        return np.allclose(sf1, sf2)


    @property
    def e(self):
        return self._ener

    @property
    def e_occ(self):
        return self.e[self.e < self.chempot]

    @property
    def e_vir(self):
        return self.e[self.e >= self.chempot]

    @property
    def v(self):
        return self._coup

    @property
    def v_occ(self):
        return self.v[:, self.e < self.chempot]

    @property
    def v_vir(self):
        return self.v[:, self.e >= self.chempot]

    @property
    def w(self):
        return util.einsum('xk,xk->k', self.v, self.v)

    @property
    def w_occ(self):
        return util.einsum('xk,xk->k', self.v_occ, self.v_occ)

    @property
    def w_vir(self):
        return util.einsum('xk,xk->k', self.v_vir, self.v_vir)

    @property
    def nphys(self):
        return self.v.shape[0]

    @property
    def naux(self):
        return self.v.shape[1]

    @property
    def nocc(self):
        return self.v_occ.shape[1]

    @property
    def nvir(self):
        return self.v_vir.shape[1]
