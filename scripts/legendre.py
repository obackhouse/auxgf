import numpy as np
import scipy.special
import scipy.integrate

try:
    import sympy as sp
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

    str_hi = str(hi)
    str_lo = str(lo)


    # Some light optimisation by removing the powers and declaring
    # them as variables avoiding recomputation, b and t should be
    # raise up to order n and a up to order n+1.

    for i in range(n+1, 1, -1):
        str_hi = str_hi.replace('**%d' % i, '%d' % i)
        str_lo = str_lo.replace('**%d' % i, '%d' % i)


    # Change the Heaviside and exp function

    str_hi = str_hi.replace('Heaviside(a)', '(a > 0)')
    str_lo = str_lo.replace('Heaviside(a)', '(a > 0)')
    str_hi = str_hi.replace('exp', 'np.exp')
    str_lo = str_lo.replace('exp', 'np.exp')


    # Write the function

    func  = 'def legendre_bath_kernel_%d(e, beta):\n' % n

    func += '    a = e\n'
    func += '    b = beta\n'

    func += '    a2 = a*a\n'
    if n > 1:
        func += '    b2 = b*b\n'
        func += '    t2 = t*t\n'
        for i in range(3, n+1):
            func += '    a%d = a%d*a\n' % (i, i-1) 
            func += '    b%d = b%d*b\n' % (i, i-1)
            func += '    t%d = t%d*t\n' % (i, i-1)
        func += '    a%d = a%d*a\n' % (n+1, n)

    func += '    hi = %s\n' % str_hi
    func += '    lo = %s\n' % str_lo
    func += '    return hi - lo\n'

    return func



















