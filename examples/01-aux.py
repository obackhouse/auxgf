# Builds a set of random auxiliaries and calls a few functions

import os
import numpy as np
from auxgf import aux
from auxgf.util import Timer

timer = Timer()


# Set the number of physical and auxiliary degrees of freedom:
nphys = 20
naux = 1000

# Randomise the energies, couplings and physical Hamiltonian:
energies = np.random.random((naux)) - 0.5
couplings = 0.1 * (np.random.random((nphys, naux)) - 0.5)
hamiltonian = np.random.random((nphys, nphys)) - 0.5
hamiltonian = hamiltonian + hamiltonian.T

# Build the auxiliary object:
se = aux.Aux(energies, couplings, chempot=0.0)

# Get the 'extended Fock matrix':
f_ext = se.as_hamiltonian(hamiltonian)

# Get the occupied (or virtual) parts of the self-energy:
se_occ = se.as_occupied()
assert np.all(se_occ.e < se.chempot)

# Directly compute a dot product between f_ext and a vector without building f_ext:
a = se.dot(hamiltonian, np.random.random((nphys+naux)))

# Save and load the object:
se.save('poles')
se = aux.Aux.load('poles')
os.remove('poles')

# Compress the auxiliaries:
se = se.compress(hamiltonian, nmom=(3,4))


print('time elapsed: %d min %.4f s' % (timer.total() // 60, timer.total() % 60))
