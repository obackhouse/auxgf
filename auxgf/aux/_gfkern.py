''' Contains generated code for computing the Legendre bath kernels
    in `auxgf.aux.gftrunc` with `method=_legendre`.
'''

import numpy as np


_max_kernel_order = 8

def _legendre_bath_kernel_1(e, beta):
    a = e
    b = beta
    x0 = a*b
    x1 = 1.0*x0
    x2 = np.exp(x0)
    val = (-x1*x2 - x1 + 2.0*x2 - 2.0)*np.exp(-x0*np.heaviside(a, 0.0))/(a**2*b)
    return val

def _legendre_bath_kernel_2(e, beta):
    a = e
    b = beta
    x0 = b**2
    x1 = a*b
    x2 = 6.0*x1
    x3 = np.exp(x1)
    x4 = 1.0*a**2*x0
    val = 1.0*(-x2*x3 - x2 + x3*x4 + 12.0*x3 - x4 - 12.0)*np.exp(-x1*np.heaviside(a, 0.0))/(a**3*x0)
    return val

def _legendre_bath_kernel_3(e, beta):
    a = e
    b = beta
    x0 = b**3
    x1 = a*b
    x2 = 60.0*x1
    x3 = np.exp(x1)
    x4 = 12.0*a**2*b**2
    x5 = 1.0*a**3*x0
    val = (-x2*x3 - x2 + x3*x4 - x3*x5 + 120.0*x3 - x4 - x5 - 120.0)*np.exp(-x1*np.heaviside(a, 0.0))/(a**4*x0)
    return val

def _legendre_bath_kernel_4(e, beta):
    a = e
    b = beta
    x0 = b**4
    x1 = np.heaviside(a, 0.0)
    x2 = a*b
    x3 = 20.0*a**3*b**3
    x4 = 840.0*x2
    x5 = 1.0*a**4*x0 + 180.0*a**2*b**2 + 1680.0
    val = ((-x3 - x4 + x5)*np.exp(x1*x2) - (x3 + x4 + x5)*np.exp(x2*(x1 - 1)))*np.exp(x2*(1 - 2*x1))/(a**5*x0)
    return val

def _legendre_bath_kernel_5(e, beta):
    a = e
    b = beta
    x0 = b**5
    x1 = np.heaviside(a, 0.0)
    x2 = a*b
    x3 = 1.0*a**5*x0
    x4 = 420.0*a**3*b**3
    x5 = 15120.0*x2
    x6 = 30.0*a**4*b**4 + 3360.0*a**2*b**2 + 30240.0
    val = ((-x3 - x4 - x5 + x6)*np.exp(x1*x2) - (x3 + x4 + x5 + x6)*np.exp(x2*(x1 - 1)))*np.exp(x2*(1 - 2*x1))/(a**6*x0)
    return val

def _legendre_bath_kernel_6(e, beta):
    a = e
    b = beta
    x0 = b**6
    x1 = np.heaviside(a, 0.0)
    x2 = a*b
    x3 = 42.0*a**5*b**5
    x4 = 10080.0*a**3*b**3
    x5 = 332640.0*x2
    x6 = 1.0*a**6*x0 + 840.0*a**4*b**4 + 75600.0*a**2*b**2 + 665280.0
    val = ((-x3 - x4 - x5 + x6)*np.exp(x1*x2) - (x3 + x4 + x5 + x6)*np.exp(x2*(x1 - 1)))*np.exp(x2*(1 - 2*x1))/(a**7*x0)
    return val

def _legendre_bath_kernel_7(e, beta):
    a = e
    b = beta
    x0 = b**7
    x1 = np.heaviside(a, 0.0)
    x2 = a*b
    x3 = 1.0*a**7*x0
    x4 = 1512.0*a**5*b**5
    x5 = 277200.0*a**3*b**3
    x6 = 8648640.0*x2
    x7 = 56.0*a**6*b**6 + 25200.0*a**4*b**4 + 1995840.0*a**2*b**2 + 17297280.0
    val = ((-x3 - x4 - x5 - x6 + x7)*np.exp(x1*x2) - (x3 + x4 + x5 + x6 + x7)*np.exp(x2*(x1 - 1)))*np.exp(x2*(1 - 2*x1))/(a**8*x0)
    return val

def _legendre_bath_kernel_8(e, beta):
    a = e
    b = beta
    x0 = b**8
    x1 = np.heaviside(a, 0.0)
    x2 = a*b
    x3 = 72.0*a**7*b**7
    x4 = 55440.0*a**5*b**5
    x5 = 8648640.0*a**3*b**3
    x6 = 259459200.0*x2
    x7 = 1.0*a**8*x0 + 2520.0*a**6*b**6 + 831600.0*a**4*b**4 + 60540480.0*a**2*b**2 + 518918400.0
    val = ((-x3 - x4 - x5 - x6 + x7)*np.exp(x1*x2) - (x3 + x4 + x5 + x6 + x7)*np.exp(x2*(x1 - 1)))*np.exp(x2*(1 - 2*x1))/(a**9*x0)
    return val

_legendre_bath_kernels = [
        _legendre_bath_kernel_1,
        _legendre_bath_kernel_2,
        _legendre_bath_kernel_3,
        _legendre_bath_kernel_4,
        _legendre_bath_kernel_5,
        _legendre_bath_kernel_6,
        _legendre_bath_kernel_7,
        _legendre_bath_kernel_8
]

def _legendre_bath_kernel(n, e, beta):
    if n > 8:
        raise ValueError('Generated code for Legendre bath orbitals '
                         'only exist up to n = 8.')

    if n == 0:
        return np.ones_like(e)

    func = _legendre_bath_kernels[n-1]

    return func(e, beta)



