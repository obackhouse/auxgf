''' ADC(2) class.
'''

import numpy as np
from pyscf import adc


class ADC2:
    ''' Class for second-order algebraic diagrammatric construction.

    Parameters
    ----------
    hf : RHF or UHF
        Hartree-Fock object

    See pyscf.adc for additional arguments

    Attributes
    ----------
    e_tot : float
        total energy
    e_corr : float
        correlation energy
    e_hf : float
        Hartree-Fock energy
    ip : float, ndarray of floats
        ionization potential
    ea : float, ndarray of floats
        electron affinity

    Methods
    -------
    run(*args, **kwargs)
        runs the calculation (performed automatically), see class
        parameters for arguments
    '''

    def __init__(self, hf, **kwargs):
        self.hf = hf
        self._pyscf = adc.ADC(self.hf._pyscf, **kwargs)

    def run(self, **kwargs):
        self._pyscf.run(**kwargs)
        return self

    @property
    def e_tot(self):
        return self._pyscf.e_tot

    @property
    def e_corr(self):
        return self.e_tot - self.e_hf

    @property
    def e_hf(self):
        return self.hf.e_tot

    @property
    def ip(self):
        return self._pyscf.ip_adc()[:2]

    @property
    def ea(self):
        return self._pyscf.ea_adc()[:2]
