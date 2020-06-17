''' Class to perform auxiliary MP2 calculations for restricted
    references.
'''

import numpy as np
import functools

from auxgf import util, aux
from auxgf.util import types, log, mpi


def _set_options(options, **kwargs):
    options.update({ 'nmom' : (None,None),
                     'wtol' : 1e-12,
                     'ss_factor' : 1.0,
                     'os_factor' : 1.0,
                     'bath_type' : 'power',
                     'bath_beta' : 100,
                     'qr' : 'cholesky',
    })

    for key,val in kwargs.items():
        if key not in options.keys():
            raise ValueError('%s argument invalid.' % key)

    options.update(kwargs)

    options['_build'] = {
        'wtol' : options['wtol'],
        'ss_factor' : options['ss_factor'],
        'os_factor' : options['os_factor'],
    }

    options['_merge'] = {
        'method' : options['bath_type'],
        'beta' : options['bath_beta'],
        'qr' : options['qr'],
    }

    return options


class RAMP2(util.AuxMethod):
    ''' Restricted auxiliary MP2 method.

    Parameters
    ----------
    rhf : RHF
        Hartree-Fock object
    nmom : tuple of int, optional
        number of moments to which the truncation is constistent to,
        ordered by (Green's function, self-energy), default is
        (None, None) which means no truncation (full MP2).
    dm0 : (n,n) ndarray, optional
        density matrix, if None, use rhf.rdm1_mo, default None
    verbose : bool, optional
        if True, print output log, default True
    wtol : float, optional
        minimum pole weight to be considered zero, default 1e-12
    ss_factor : float, optional
        same spin factor for auxiliary build, default 1.0
    os_factor : float, optional
        opposite spin factor for auxiliary build, default 1.0
    bath_type : str, optional
        GF truncation kernel method {'power', 'legendre'}, default 
        'power'
    bath_beta : int, optional
        inverse temperature used in GF truncation kernel, default 100
    qr : str, optional
        type of QR solver to use for SE truncation {'cholesky', 
        'numpy', 'scipy', 'unsafe'}, default 'cholesky'

    Attributes
    ----------
    hf : RHF
        Hartree-Fock object
    nmom : tuple of int
        see parameters
    verbose : bool
        see parameters
    options : dict
        dictionary of options
    rdm1 : (n,n) ndarray
        one-particle reduced density matrix, projected into the
        physical basis
    se : Aux
        auxiliary representation of the self-energy
    e_1body : float
        Hartree-Fock energy
    e_2body : float
        MP2 energy 
    e_tot : float
        total energy
    e_mp2 : float
        MP2 energy 
    e_corr : float
        correlation energy i.e. `e_tot` - `hf.e_tot`

    Methods
    -------
    setup(rhf)
        constructs the object using the parameters provided to 
        `__init__` 
    get_fock(rdm1=None)
        returns the Fock matrix resulting from the current, or
        provided, density
    run()
        runs the method
    '''

    def __init__(self, rhf, **kwargs):
        super().__init__(rhf, **kwargs)

        self.options = _set_options(self.options, **kwargs)

        self.setup()


    @util.record_time('setup')
    def setup(self):
        super().setup()

        log.title('Options', self.verbose)
        log.options(self.options, self.verbose)
        log.title('Input', self.verbose)
        log.molecule(self.hf.mol, self.verbose)
        log.write('Basis = %s\n' % self.hf.mol.basis, self.verbose)
        log.write('E(nuc) = %.12f\n' % self.hf.e_nuc, self.verbose)
        log.write('E(hf)  = %.12f\n' % self.hf.e_tot, self.verbose)
        log.write('nao = %d\n' % self.hf.nao, self.verbose)
        log.write('nfrozen = (%d, %d)\n' % self.options['frozen'], self.verbose)
        log.write('nmom = (%s, %s)\n' % self.nmom, self.verbose)


    @util.record_time('build')
    def build(self):
        eri = self.eri
        self.se = aux.build_rmp2(self.hf.e, self.eri, **self.options['_build'],
                                 chempot=self.chempot)

        log.write('naux (build) = %d\n' % self.naux, self.verbose)


    @util.record_time('merge')
    def merge(self):
        nmom_gf, nmom_se = self.nmom

        if nmom_gf is None and nmom_se is None:
            return

        self.se = self.se.compress(self.get_fock(), self.nmom, 
                                   **self.options['_merge'])

        log.write('naux (merge) = %d\n' % self.naux, self.verbose)


    @util.record_time('energy')
    @util.record_energy('mp2')
    def energy_mp2(self):
        emp2 = aux.energy.energy_mp2_aux(self.hf.e, self.se)

        log.write('E(mp2) = %.12f\n' % emp2, self.verbose)

        return emp2


    def run(self):
        self.build()
        self.merge()
        self.energy_mp2()

        self._timings['total'] = self._timings.get('total', 0.0) \
                                 + self._timer.total()
        log.title('Timings', self.verbose)
        log.timings(self._timings, self.verbose)

        return self


    @property
    def e_1body(self):
        return self.e_hf

    @property
    def e_2body(self):
        return self.e_mp2

    @property
    def e_tot(self):
        return self.e_hf + self.e_mp2

    @property
    def e_mp2(self):
        return self._energies['mp2'][-1]


    @property
    def nmom(self):
        return self.options['nmom']

    @nmom.setter
    def nmom(self, val):
        self.options['nmom'] = val
