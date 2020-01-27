''' Base class for grids.
'''

import numpy as np

from auxgf.util import types


attr = ['dtype', 'buffer', 'offset', 'strides', 'order']

class Grid(np.ndarray):
    ''' Base class for a grid, inherits numpy.ndarray.

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
    beta : int, optional
        inverse temperature (default 256)
    lamb : float, optional
        Legendre quadrature diffuse factor (default 1.0)
    prefac : float or complex, optional
        prefactor of grid values (default 1.0)

    Attributes
    ----------
    weights : (n) ndarray
        weighting factors
    values : (n) ndarray
        grid values
    type : str
        grid type {'grid', 'quad'}
    axis : str
        grid axis {'imag', 'real'}
    quantity : str
        grid quantity {'fq', 'tm'}

    Methods
    -------
    __getitem__(key)
        indexing of the values using numpy indexing
    '''

    def __new__(self, npts, **kwargs):
        numpy_flags = { k:v for k,v in kwargs.items() if k in attr }
        return super().__new__(self, npts, **numpy_flags)

    def __init__(self, npts, **kwargs):
        self.npts = npts

        for item in kwargs.items():
            setattr(self, *item)

    @property
    def type(self):
        return self.__class__.__name__[-4:].lower()

    @property
    def axis(self):
        axis = self.__class__.__name__[:2].lower()

        if axis == 're':
            return 'real'
        elif axis == 'im':
            return 'imag'
        else:
            raise ValueError

    def quantity(self):
        quant = self.__class__.__name__[2:4].lower()

        if axis == 'fq':
            return 'freq'
        elif axis == 'tm':
            return 'time'
        else:
            raise ValueError

    @property
    def weights(self):
        return self._wts

    @weights.setter
    def weights(self, vals):
        self._wts = vals

    @property
    def values(self):
        return self[:]

    @values.setter
    def values(self, vals):
        self[:] = vals

    def copy(self):
        return self.__class__(self.npts,
                              eta=self.eta,
                              beta=self.beta,
                              lamb=self.lamb,
                              minpt=self.minpt,
                              maxpt=self.maxpt)

