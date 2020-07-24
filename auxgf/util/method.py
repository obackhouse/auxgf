''' Base class for a general auxgf auxiliary method.
'''

import numpy as np

from auxgf import util, aux
from auxgf.util import types, mpi


class AuxMethod:
    ''' This is a base class for most of the auxiliary methods
        implemented in auxgf. There are some methods and properties
        implemented which can be overwritten in inherited classes:
    
        The following methods are always defined:
        
            __init__
            setup
            run
            get_fock
            ip (property)
            ea (property)

        The following options are always set, with default values:

            dm0 : None
            frozen : 0
            verbose : True

        The following properties are inherited from self.hf:

            nalph
            nbeta
            nelec
            e_hf

        The following properties are inherited from self.se:

            nphys
            naux
            chempot

        The following properties are inherited from self._energies:

            e_1body
            e_2body
            e_tot
            e_corr

        The follow properties are inherited from self.options:

            verbose

        In each instance additional options should be set in __init__,
        and then setup must be called manually. By default timings are
        not recorded for the setup function.

        All methods contain _timings, _energies, converged and
        iteration flags. Some of these may be non-applicable in 
        some methods, for which they should be overwritten with a
        @property call leading to an error.
    '''

    def __init__(self, hf, **kwargs):
        self.hf = hf
        self._timer = util.Timer()

        self.options = {}
        self.options['dm0'] = None
        self.options['frozen'] = (0, 0)
        self.options['verbose'] = mpi.rank == 0


    def setup(self):
        self.h1e = self.hf.h1e_mo
        self.eri = self.hf.eri_mo

        unrestricted = self.h1e.ndim == 3

        if self.options['dm0'] is None:
            self.rdm1 = self.hf.rdm1_mo
        else:
            self.rdm1 = np.array(self.options['dm0'], dtype=types.float64)

            if unrestricted and self.rdm1.ndim == 2:
                self.rdm1 = np.stack([self.rdm1, self.rdm1], axis=0)

        self.converged = False
        self.iteration = 0

        nact = self.hf.nao - sum(self.options['frozen'])

        if not unrestricted:
            self.se = aux.Aux([], [[],]*nact, chempot=self.hf.chempot)
            self.gf = self.se.new(self.hf.e, np.eye(nact))
            self._se_prev = None
        else:
            self.se = (aux.Aux([], [[],]*nact, chempot=self.hf.chempot[0]),
                       aux.Aux([], [[],]*nact, chempot=self.hf.chempot[1]))
            self.gf = (self.se[0].new(self.hf.e[0], np.eye(nact)),
                       self.se[1].new(self.hf.e[1], np.eye(nact)))
            self._se_prev = (None, None)

        self._timings = {}
        self._energies = {}
        

    def get_fock(self, rdm1=None):
        if rdm1 is None:
            rdm1 = self.rdm1

        fock = self.hf.get_fock(rdm1, basis='mo')

        return fock


    def get_fock_act(self, rdm1=None):
        c, v = self.options['frozen']
        arg = slice(c, -v if v else None)

        return  self.get_fock(rdm1=rdm1)[...,arg,arg]


    def get_eri_act(self):
        c, v = self.options['frozen']
        arg = slice(c, -v if v else None)

        if self.hf.with_df:
            return self.eri[...,arg,arg]
        else:
            return self.eri[...,arg,arg,arg,arg]


    def solve_dyson(self):
        def _solve_dyson(se, fock):
            e, c = se.eig(fock)
            c = c[:se.nphys]
            gf = se.new(e, c)
            return gf

        fock = self.get_fock_act()
            
        if isinstance(self.se, (list, tuple)):
            assert fock.shape[0] == len(self.se)
            self.gf = tuple([_solve_dyson(se, f) for se,f in zip(self.se, fock)])
        else:
            self.gf = _solve_dyson(self.se, fock)


    @property
    def ip(self):
        e, v = -np.inf, None

        if isinstance(self.gf, (list, tuple)):
            for gf in self.gf:
                gf_occ = gf.as_occupied()
                arg = np.argmax(gf_occ.e)

                if gf_occ.e[arg] > e:
                    e, v = gf_occ.e[arg], gf_occ.v[:,arg]

        else:
            gf_occ = self.gf.as_occupied()
            arg = np.argmax(gf_occ.e)
            e, v = gf_occ.e[arg], gf_occ.v[:,arg]

        return -e, v


    @property
    def ea(self):
        e, v = np.inf, None

        if isinstance(self.gf, (list, tuple)):
            for gf in self.gf:
                gf_vir = gf.as_virtual()
                arg = np.argmin(gf_vir.e)

                if gf_vir.e[arg] < e:
                    e, v = gf_vir.e[arg], gf_vir.v[:,arg]

        else:
            gf_vir = self.gf.as_virtual()
            arg = np.argmin(gf_vir.e)
            e, v = gf_vir.e[arg], gf_vir.v[:,arg]

        return e, v


    def run(self):
        raise NotImplementedError


    @property
    def nalph(self):
        return self.hf.nalph

    @property
    def nbeta(self):
        return self.hf.nbeta

    @property
    def nelec(self):
        return self.hf.nelec

    @property
    def e_hf(self):
        return self.hf.e_tot


    @property
    def nphys(self):
        if isinstance(self.se, (tuple, list)):
            return self.se[0].nphys
        else:
            return self.se.nphys

    @property
    def naux(self):
        if isinstance(self.se, (tuple, list)):
            return tuple([x.naux for x in self.se])
        else:
            return self.se.naux

    @property
    def chempot(self):
        if isinstance(self.se, (tuple, list)):
            return tuple([x.chempot for x in self.se])
        else:
            return self.se.chempot

    @property
    def e_1body(self):
        if len(self._energies.get('1b', [])):
            return self._energies['1b'][-1]
        else:
            return self.e_hf

    @property
    def e_2body(self):
        if len(self._energies.get('2b', [])):
            return self._energies['2b'][-1]
        else:
            raise NotImplementedError

    @property
    def e_tot(self):
        if len(self._energies.get('tot', [])):
            return self._energies['tot'][-1]
        else:
            return self.e_1body + self.e_2body

    @property
    def e_corr(self):
        return self.e_tot - self.e_hf


    @property
    def verbose(self):
        return self.options['verbose']

    @verbose.setter
    def verbose(self, val):
        self.options['verbose'] = val
