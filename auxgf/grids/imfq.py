''' Imaginary frequency grid classes.
'''

import numpy as np
import scipy.special

from auxgf.util import types
from auxgf.grids import Grid


class ImFqGrid(Grid):
    ''' Class for a uniform imaginary-frequency grid.

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
        
        self.minpt = 0.5 * self.d
        self.maxpt = (npts - 0.5) * self.d

        self.values = np.linspace(self.minpt, self.maxpt, self.npts)
        self.weights = None

    @property
    def d(self):
        return 2.0 * np.pi / self.beta


class ImFqQuad(Grid):
    ''' Class for a Legndre-quadrature imaginary-frequency grid.

    Parameters
    ----------
    npts : int
        number of grid points
    lamb : float, optional
        Legendre quadrature diffuse factor (default 1.0)
        
    Attributes
    ----------
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
        
        x, w = scipy.special.roots_legendre(npts)

        self.values = (1.0 - x) / (self.lamb * (1.0 + x))
        self.weights = w.astype(types.float64)


