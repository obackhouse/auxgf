''' ADC(2)-x class.
'''

import numpy as np
from auxgf.adc import ADC2


class ADC2x(ADC2):
    ''' Class for extended second-order algebraic diagrammatric 
        construction.

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
        super().__init__(hf, **kwargs)
        self._pyscf.method = 'adc(2)-x'
