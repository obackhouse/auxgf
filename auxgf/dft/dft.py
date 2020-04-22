''' Base density-functional theory class and routines.
'''

import numpy as np
from pyscf import dft, lib

from auxgf import util
from auxgf.util import log, types
from auxgf.hf import hf

class DFT(hf.HF):
    ''' Base class for Kohn-Sham density-functional theory.

    Parameters
    ----------
    mol : Molecule
        object defining the molecule
    xc : str, optional
        Exchange-correlation functional (default 'pbe')
    method : str, optional
        DFT types {'rdft', 'udft'} (default 'rhf')
    disable_omp : bool, optional
        disable OpenMP parallelism (default True)
    
    See pyscf.dft.rdft.KohnShamDFT for additional keyword arguments

    Attributes
    ----------

    
    Methods
    -------


    '''

    def __init__(self, mol, **kwargs):
        kwargs['disable_omp'] = False
        kwargs['check_stability'] = False
        kwargs['stability_cycles'] = 0
        kwargs['with_df'] = False
        kwargs['method'] = kwargs.get('method', dft.RKS)

        super().__init__(mol, **kwargs)

