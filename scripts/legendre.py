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
    a = sp.symbols('a', nonzero=True, real=True)
    b = sp.symbols('b', positive=True, nonzero=True, real=True)

    pn = sp.legendre(n, 2.0 * t / b + 1.0)
    gf = sp.exp(-a * (t + sp.Heaviside(a) * b))

    expr = sp.integrate(pn * gf, t)

    return expr


def codegen_legendre_bath_kernel(n):
    ''' Generates the Python code to compute the Legendre bath
        orbital for a given order.

    Parameters
    ----------
    n : int
        order of Legendre polynomial

    Returns
    -------
    func : str
        string containing the code. The first line is given as
        `def legendre_bath_kernel_n(e, beta):`, where `n` is
        replaced with the input variable.
    '''

    t = sp.symbols('t', real=True)
    a = sp.symbols('a', nonzero=True, real=True)
    b = sp.symbols('b', positive=True, nonzero=True, real=True)

    expr = int_legendre_bath_kernel(n)
    expr = sp.simplify(expr)
    hi = sp.simplify(expr.subs({t:0}))
    lo = sp.simplify(expr.subs({t:-b}))
    expr = sp.simplify(hi - lo)
    expr = sp.cse(expr)

    numpy_printer()._kf['Heaviside'] = 'numpy.heaviside'

    func  = 'def legendre_bath_kernel_%d(e, beta):\n' % n
    func += '    a = e\n'
    func += '    b = beta\n'

    for subexpr in expr[0]:
        func += '    %s\n' % numpy_printer().doprint(subexpr[1], subexpr[0])

    func += '    %s\n' % numpy_printer().doprint(expr[-1][0], 'val')
    func += '    return val\n'

    func = func.replace('numpy.heaviside(a)', 'numpy.heaviside(a, 0.0)')
    func = func.replace('numpy', 'np')

    return func


#import sys
#with open('code.dat', 'w') as sys.stdout:
#    for n in range(1, 9):
#        print(codegen_legendre_bath_kernel(n))













