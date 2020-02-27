import numpy as np
import scipy.special
import scipy.integrate

try:
    import sympy as sp
    from sympy.printing.pycode import NumPyPrinter as numpy_printer
except ImportError:
    pass


def int_legendre_bath_kernel(n):
    ''' Generates an analytical expression for the integration over
        the product between Legendre polynomial Pn(x) and the time
        Green's function expression, for use in generating Legendre
        bath orbitals.

    Parameters
    ----------
    n : int
        order of Legendre polynomial

    Returns
    -------
    expr : sympy.Expr
        sympy expression
    '''

    t = sp.symbols('t', real=True)
    a = sp.symbols('e', nonzero=True, real=True)
    b = sp.symbols('beta', positive=True, nonzero=True, real=True)

    pn = sp.legendre(n, 2.0 * t / b + 1.0)
    gf = sp.exp(-a * (t + sp.Heaviside(a) * b))

    expr = sp.integrate(pn * gf, t)

    return expr


def codegen_legendre_bath_kernel(n):
    ''' Generates the Python code to compute the Legendre bath
        orbital for a given order. The code is somewhat optimised
        with common subexpression elimination, but sympy isn't
        great at identifying CSEs with the high-order exponents.

    Parameters
    ----------
    n : int
        order of Legendre polynomial

    Returns
    -------
    func : str
        string containing the code. The first line is given as
        `def legendre_bath_kernel_n(e, beta, chempot=0.0):`, where 
        `n` is replaced with the input variable.
    '''

    t = sp.symbols('t', real=True)
    a = sp.symbols('e', nonzero=True, real=True)
    b = sp.symbols('beta', positive=True, nonzero=True, real=True)
    exp = sp.symbols('exp')
    exp_hi = sp.symbols('exp_hi')
    exp_lo = sp.symbols('exp_lo')

    exponent = sp.exp(-a * (t + b * sp.Heaviside(a)))

    expr = int_legendre_bath_kernel(n)
    expr = expr.subs({exponent: exp})
    hi = sp.simplify(expr.subs({t: 0, exp: exp_hi}))
    lo = sp.simplify(expr.subs({t: -b, exp: exp_lo}))

    expr = hi - lo
    expr = sp.cse(expr, optimizations='basic')

    numpy_printer()._kf['Heaviside'] = 'numpy.heaviside'

    func  = 'def legendre_bath_kernel_%d(e, beta, chempot=0.0):\n' % n
    func += '    e = np.asarray(e)\n'

    for subexpr in expr[0]:
        func += '    %s\n' % numpy_printer().doprint(subexpr[1], subexpr[0])

    func += '    exp_hi = np.ones_like(e)\n'
    func += '    exp_lo = np.ones_like(e)\n'
    func += '    exp_hi[e > chempot] = np.exp(-e[e > chempot]*beta)\n'
    func += '    exp_lo[e < chempot] = np.exp(e[e < chempot]*beta)\n'

    func += '    %s\n' % numpy_printer().doprint(expr[-1][0], 'val')
    func += '    return val\n'

    func = func.replace('numpy.heaviside(a)', 'numpy.heaviside(a-chempot, 0.0)')
    func = func.replace('numpy', 'np')

    return func


#print(codegen_legendre_bath_kernel(3))
#import sys
#with open('code.dat', 'w') as sys.stdout:
#    for n in range(1, 9):
#        print(codegen_legendre_bath_kernel(n))













