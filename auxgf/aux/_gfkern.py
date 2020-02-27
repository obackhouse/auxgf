''' Contains generated code for computing the Legendre bath kernels
    in `auxgf.aux.gftrunc` with `method=_legendre`.
'''

import numpy as np


_max_kernel_order = 8

def _legendre_bath_kernel_1(e, beta, chempot=0.0):
    e = np.asarray(e)
    x0 = e**(-1.0)
    x1 = 2.0*x0/beta
    exp_hi = np.ones_like(e)
    exp_lo = np.ones_like(e)
    exp_hi[e > chempot] = np.exp(-e[e > chempot]*beta)
    exp_lo[e < chempot] = np.exp(e[e < chempot]*beta)
    val = x0*(-exp_hi*x1 - 1.0*exp_hi + exp_lo*x1 - 1.0*exp_lo)
    return val

def _legendre_bath_kernel_2(e, beta, chempot=0.0):
    e = np.asarray(e)
    x0 = e**(-1.0)
    x1 = 12.0/(beta**2*e**2)
    x2 = 6.0*x0/beta
    exp_hi = np.ones_like(e)
    exp_lo = np.ones_like(e)
    exp_hi[e > chempot] = np.exp(-e[e > chempot]*beta)
    exp_lo[e < chempot] = np.exp(e[e < chempot]*beta)
    val = x0*(-exp_hi*x1 - exp_hi*x2 - 1.0*exp_hi + exp_lo*x1 - exp_lo*x2 + 1.0*exp_lo)
    return val

def _legendre_bath_kernel_3(e, beta, chempot=0.0):
    e = np.asarray(e)
    x0 = beta**3
    x1 = 12.0*beta**2*e**2
    x2 = 60.0*beta*e + 1.0*e**3*x0
    exp_hi = np.ones_like(e)
    exp_lo = np.ones_like(e)
    exp_hi[e > chempot] = np.exp(-e[e > chempot]*beta)
    exp_lo[e < chempot] = np.exp(e[e < chempot]*beta)
    val = -1.0*(exp_hi*(x1 + x2 + 120.0) + exp_lo*(-x1 + x2 - 120.0))/(e**4*x0)
    return val

def _legendre_bath_kernel_4(e, beta, chempot=0.0):
    e = np.asarray(e)
    x0 = beta**4
    x1 = 20.0*beta**3*e**3
    x2 = 840.0*beta*e
    x3 = 180.0*beta**2*e**2 + 1.0*e**4*x0 + 1680.0
    exp_hi = np.ones_like(e)
    exp_lo = np.ones_like(e)
    exp_hi[e > chempot] = np.exp(-e[e > chempot]*beta)
    exp_lo[e < chempot] = np.exp(e[e < chempot]*beta)
    val = 1.0*(-exp_hi*(x1 + x2 + x3) + exp_lo*(-x1 - x2 + x3))/(e**5*x0)
    return val

def _legendre_bath_kernel_5(e, beta, chempot=0.0):
    e = np.asarray(e)
    x0 = beta**5
    x1 = 30.0*beta**4*e**4
    x2 = 3360.0*beta**2*e**2
    x3 = 420.0*beta**3*e**3 + 15120.0*beta*e + 1.0*e**5*x0
    exp_hi = np.ones_like(e)
    exp_lo = np.ones_like(e)
    exp_hi[e > chempot] = np.exp(-e[e > chempot]*beta)
    exp_lo[e < chempot] = np.exp(e[e < chempot]*beta)
    val = -1.0*(exp_hi*(x1 + x2 + x3 + 30240.0) + exp_lo*(-x1 - x2 + x3 - 30240.0))/(e**6*x0)
    return val

def _legendre_bath_kernel_6(e, beta, chempot=0.0):
    e = np.asarray(e)
    x0 = beta**6
    x1 = 42.0*beta**5*e**5
    x2 = 10080.0*beta**3*e**3
    x3 = 332640.0*beta*e
    x4 = 840.0*beta**4*e**4 + 75600.0*beta**2*e**2 + 1.0*e**6*x0 + 665280.0
    exp_hi = np.ones_like(e)
    exp_lo = np.ones_like(e)
    exp_hi[e > chempot] = np.exp(-e[e > chempot]*beta)
    exp_lo[e < chempot] = np.exp(e[e < chempot]*beta)
    val = 1.0*(-exp_hi*(x1 + x2 + x3 + x4) + exp_lo*(-x1 - x2 - x3 + x4))/(e**7*x0)
    return val

def _legendre_bath_kernel_7(e, beta, chempot=0.0):
    e = np.asarray(e)
    x0 = beta**7
    x1 = 56.0*beta**6*e**6
    x2 = 25200.0*beta**4*e**4
    x3 = 1995840.0*beta**2*e**2
    x4 = 1512.0*beta**5*e**5 + 277200.0*beta**3*e**3 + 8648640.0*beta*e + 1.0*e**7*x0
    exp_hi = np.ones_like(e)
    exp_lo = np.ones_like(e)
    exp_hi[e > chempot] = np.exp(-e[e > chempot]*beta)
    exp_lo[e < chempot] = np.exp(e[e < chempot]*beta)
    val = -1.0*(exp_hi*(x1 + x2 + x3 + x4 + 17297280.0) + exp_lo*(-x1 - x2 - x3 + x4 - 17297280.0))/(e**8*x0)
    return val

def _legendre_bath_kernel_8(e, beta, chempot=0.0):
    e = np.asarray(e)
    x0 = beta**8
    x1 = 72.0*beta**7*e**7
    x2 = 55440.0*beta**5*e**5
    x3 = 8648640.0*beta**3*e**3
    x4 = 259459200.0*beta*e
    x5 = 2520.0*beta**6*e**6 + 831600.0*beta**4*e**4 + 60540480.0*beta**2*e**2 + 1.0*e**8*x0 + 518918400.0
    exp_hi = np.ones_like(e)
    exp_lo = np.ones_like(e)
    exp_hi[e > chempot] = np.exp(-e[e > chempot]*beta)
    exp_lo[e < chempot] = np.exp(e[e < chempot]*beta)
    val = 1.0*(-exp_hi*(x1 + x2 + x3 + x4 + x5) + exp_lo*(-x1 - x2 - x3 - x4 + x5))/(e**9*x0)
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

def _legendre_bath_kernel(n, e, beta, chempot=0.0):
    if n > 8:
        raise ValueError('Generated code for Legendre bath orbitals '
                         'only exist up to n = 8.')

    if n == 0:
        return np.ones_like(e)

    func = _legendre_bath_kernels[n-1]

    return func(e, beta, chempot=chempot)



