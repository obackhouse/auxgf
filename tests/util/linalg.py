import numpy as np

from auxgf.util import linalg

np.random.seed = 12345


array_a = np.full((2, 2), fill_value=1)
array_b = np.full((2, 3), fill_value=2)
ref = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 2, 2, 2], [0, 0, 2, 2, 2]])
assert np.all(ref == linalg.block_diag([array_a, array_b]))


array = np.zeros((5,5), dtype=int)
diag = linalg.diagonal(array)
diag += 1
assert np.all(array == np.eye(5, dtype=int))


#array = np.outer(*[np.random.random(5),]*2)
#t, v = linalg.lanczos(array)
#np.set_printoptions(precision=3)
#assert np.allclose(np.dot(np.dot(v, t), v.T), array)


not_herm = np.random.random((5, 5))
herm = not_herm + not_herm.T
assert not linalg.is_hermitian(not_herm) and linalg.is_hermitian(herm)


vecs = [np.random.random((3)) for x in range(3)]
array = linalg.outer_sum(vecs)
ref = vecs[0][:,None,None] + vecs[1][None,:,None] + vecs[2][None,None,:]
assert np.allclose(array, ref)


array = linalg.outer_mul(vecs)
ref = vecs[0][:,None,None] * vecs[1][None,:,None] * vecs[2][None,None,:]
assert np.allclose(array, ref)


arrays = [np.random.random((4,4)) for x in range(4)] + [np.random.random((5,5)) for x in range(3)]
ws, vs = linalg.batch_eigh(arrays)
wref, vref = zip(*[linalg.eigh(x) for x in arrays])
assert np.all([np.allclose(a, b) for a,b in zip(ws, wref)]) and np.all([np.allclose(a, b) for a,b in zip(vs, vref)])


#ij = np.random.random((4,5))
#jk = np.random.random((5,6))
#array = linalg.dirsum('ij,jk->ijk', ij, jk)
#ref = np.array([[[ij[i,j] + jk[j,k] for k in range(6)] for j in range(5)] for i in range(4)])
#assert np.allclose(array, ref)


array = np.random.random((4,4))
assert np.allclose(np.max(array), linalg.amax(array))
assert np.allclose(np.min(array), linalg.amin(array))
array = np.array([[]])
assert np.isnan(linalg.amax(array))
assert np.isnan(linalg.amin(array))

