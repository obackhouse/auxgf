''' Direct inversion of iterative subspace (Pulay method).
'''

import numpy as np
from pyscf import lib


class DIIS(lib.diis.DIIS):
    ''' Class to control DIIS (essentially a wrapper to 
        pyscf.lib.diis.DIIS).

    Parameters
    ----------
    space : int
        number of DIIS vectors to store
    min_space : int
        minimum number of DIIS vectors before extrapolation

    Methods
    -------
    update(x, xerr=None):
        push the error vectors and then extrapolate a vector via DIIS,
        if xerr=None then the difference between subsequent vectors
        is used
    '''

    def __init__(self, space, min_space=1):
        super().__init__()
        self.space = space
        self.min_space = min_space

    def update(self, x, xerr=None):
        try:
            return super().update(x, xerr=xerr)
        except np.linalg.linalg.LinAlgError:
            return x



