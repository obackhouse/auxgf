''' MP2 class.
'''

import numpy as np
from pyscf import mp


class MP2:
    ''' Class for second-order Moller-Plesset perturbation theory.

    Parameters
    ----------
    hf : RHF, UHF, ROHF or GHF
        Hartree-Fock object
    
    See pyscf.mp.mp2.MP2 for additional arguments

    Attributes
    ----------
    e_tot : float
        total energy
    e_corr : float
        correlation energy
    e_hf : float
        Hartree-Fock energy

    Methods
    -------
    run(*args, **kwargs)
        runs the calculation (performed automatically), see class
        parameters for arguments.
    '''

    def __init__(self, hf, *args, **kwargs):
        self.hf = hf
        self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        mp2 = mp.MP2(self.hf._pyscf, *args, **kwargs)
        mp2.run()

        self._pyscf = mp2

    @property
    def e_tot(self):
        return self._pyscf.e_tot

    @property
    def e_corr(self):
        return self.e_tot - self.e_hf

    @property
    def e_hf(self):
        return self.hf.e_tot
