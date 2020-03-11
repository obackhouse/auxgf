''' CCSD class.
'''

import numpy as np
from pyscf import cc


class CCSD:
    ''' Class for coupled cluster singles and doubles.

    Parameters
    ----------
    hf : RHF, UHF, ROHF or GHF
        Hartree-Fock object

    See pyscf.cc.ccsd.CCSD for additional arguments

    Attributes
    ----------
    e_tot : float
        total energy
    e_corr : float
        correlation energy
    e_hf : float
        Hartree-Fock energy
    e_t : float
        perturbative triples energy

    Methods
    -------
    run(*args, **kwargs)
        runs the calculation (performed automatically), see class
        parameters for arguments
    '''

    def __init__(self, hf, *args, **kwargs):
        self.hf = hf
        self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        ccsd = cc.CCSD(self.hf._pyscf, *args, **kwargs)
        ccsd.run()

        self._pyscf = ccsd

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
    def e_t(self):
        return self._pyscf.ccsd_t()
