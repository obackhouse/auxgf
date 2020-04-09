# Builds a set of auxiliaries representing an RMP2 self-energy and calculates the correlation energy

import numpy as np
from auxgf import mol, hf, aux, grids, mp
from auxgf.util import Timer

timer = Timer()


# Build the Molecule object:
m = mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='cc-pvdz')

# Build the RHF object:
rhf = hf.RHF(m)
rhf.run()

# Build some MO quantities:
e_mo = rhf.e
eri_mo = rhf.eri_mo
fock_mo = rhf.fock_mo
chempot = rhf.chempot

# We can build the auxiliaries via the MO energies and MO integrals:
se_rmp2_a = aux.build_rmp2(e_mo, eri_mo, chempot=chempot)

# We can also do this by iterating (once) an empty auxiliary space:
s0 = aux.Aux(np.zeros(0), np.zeros((rhf.nao,0)), chempot=chempot)
se_rmp2_b = aux.build_rmp2_iter(s0, fock_mo, eri_mo) 
# In the above, chempot is inherited from s0

# Calculate the MP2 energies and compare them to canonical MP2:
e_mp2_a = aux.energy_mp2(rhf.e, se_rmp2_a)
e_mp2_b = aux.energy_mp2(rhf.e, se_rmp2_b)
e_mp2_c = mp.MP2(rhf).run().e_corr
print('a) E(mp2) = %.12f' % e_mp2_a)
print('b) E(mp2) = %.12f' % e_mp2_b)
print('c) E(mp2) = %.12f' % e_mp2_c)

# a) and c) should be exact, because they use the same eigenvalue and eigenvectors
# to sum over. b) will be a little different, because the iteration rediagonalises
# the physical-space Fock matrix - but should be very close.


print('time elapsed: %d min %.4f s' % (timer.total() // 60, timer.total() % 60))
