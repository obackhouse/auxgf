''' Imaginary time grid classes.
'''

import numpy as np
import scipy.special

from auxgf.util import types
from auxgf.grids import Grid


class ImTmGrid(Grid):
    ''' Class for a uniform imaginary-time grid.

    Parameters
    ----------
    npts : int
        number of grid points
    beta : int, optional
        inverse temperature (default 256)

    Attributes
    ----------
    d : float
        separation between points = 2 pi / beta

    See auxgf.grids.grid

    Methods
    -------
    See auxgf.grids.grid
    '''

    def __new__(self, npts, **kwargs):
        return super().__new__(self, npts, **kwargs)

    def __init__(self, npts, **kwargs):
        super().__init__(npts, **kwargs)

        self.prefac = 1.0j
        self.eta = 0.0

        self.minpt = -self.beta + 0.5 * self.d
        self.maxpt = -0.5 * self.d

        self.values = np.linspace(self.minpt, self.maxpt, self.npts)
        self.weights = None

    @property
    def d(self):
        return 2.0 * np.pi / self.beta
