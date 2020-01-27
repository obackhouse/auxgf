import numpy as np

from auxgf import util


nold = 5
nnew = 4

rand = np.random.random


array = rand((nold, nold))
c_a = rand((nold, nnew))
c_b = rand((nold, nnew))
t1 = util.ao2mo(array, c_a, c_b)
t2 = np.einsum('pq,pi,qj->ij', array, c_a, c_b)
assert np.allclose(t1, t2)

array = rand((nold, nold))
c_a = rand((2, nold, nnew))
c_b = rand((2, nold, nnew))
t1 = util.ao2mo(array, c_a, c_b)
t2 = np.einsum('pq,spi,sqj->sij', array, c_a, c_b)
assert np.allclose(t1, t2)

array = rand((2, nold, nold))
c_a = rand((2, nold, nnew))
c_b = rand((2, nold, nnew))
t1 = util.ao2mo(array, c_a, c_b)
t2 = np.einsum('spq,spi,sqj->sij', array, c_a, c_b)
assert np.allclose(t1, t2)

array = rand((nold, nold, nold, nold))
c_a = rand((nold, nnew))
c_b = rand((nold, nnew))
c_c = rand((nold, nnew))
c_d = rand((nold, nnew))
t1 = util.ao2mo(array, c_a, c_b, c_c, c_d)
t2 = np.einsum('pqrs,pi,qj,rk,sl->ijkl', array, c_a, c_b, c_c, c_d)
assert np.allclose(t1, t2)

array = rand((nold, nold, nold, nold))
c_a = rand((2, nold, nnew))
c_b = rand((2, nold, nnew))
c_c = rand((2, nold, nnew))
c_d = rand((2, nold, nnew))
t1 = util.ao2mo(array, c_a, c_b, c_c, c_d)
t2 = np.einsum('pqrs,api,aqj,brk,bsl->abijkl', array, c_a, c_b, c_c, c_d)
assert np.allclose(t1, t2)

array = rand((2, 2, nold, nold, nold, nold))
c_a = rand((2, nold, nnew))
c_b = rand((2, nold, nnew))
c_c = rand((2, nold, nnew))
c_d = rand((2, nold, nnew))
t1 = util.ao2mo(array, c_a, c_b, c_c, c_d)
t2 = np.einsum('abpqrs,api,aqj,brk,bsl->abijkl', array, c_a, c_b, c_c, c_d)
assert np.allclose(t1, t2)


array = rand((nold, nold, nold, nold))
c_a = rand((nold, nnew))
c_b = rand((nold, nnew))
c_c = rand((nold, nnew))
t1 = util.mo2qo(array, c_a, c_b, c_c)
t2 = np.einsum('pqrs,qi,rj,sk->pijk', array, c_a, c_b, c_c)
assert np.allclose(t1, t2)


#TODO: test SemiDirectMO2QO








