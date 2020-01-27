''' Real frequency grid classes.
'''

import numpy as np
import scipy.special

from auxgf.util import types
from auxgf.grids import Grid

class ReFqGrid(Grid):
    ''' Class for a uniform real-frequency grid.

    Parameters
    ----------
    npts : int
        number of grid points
    minpt : float, optional
        value of minimum grid point (default -5.0)
    maxpt : float, optional
        value of maximum grid point (default 5.0)
    eta : float, optional
        broadening factor for imaginary axis (default 0.1)

    Attributes
    ----------
    d : float
        separation between points = (maxpt - minpt) / npts

    See auxgf.grids.grid

    Methods
    -------
    See auxgf.grids.grid
    '''

    def __new__(self, npts, **kwargs):
        return super().__new__(self, npts, **kwargs)

    def __init__(self, npts, **kwargs):
        super().__init__(npts, **kwargs)

        self.prefac = 1.0

        self.values = np.linspace(self.minpt, self.maxpt, self.npts)
        self.weights = None

    @property
    def d(self):
        return (self.maxpt - self.minpt) / self.npts
