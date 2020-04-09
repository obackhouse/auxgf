# Builds a set of auxiliaries representing an UMP2 self-energy and calculates the correlation energy

import numpy as np
from auxgf import mol, hf, aux, grids, mp
from auxgf.util import Timer

timer = Timer()


# Build the Molecule object:
m = mol.Molecule(atoms='H 0 0 0; Be 0 0 1', basis='cc-pvdz', spin=1)

# Build the RHF object:
uhf = hf.UHF(m)
uhf.run()

# Build some MO quantities:
e_mo = uhf.e
eri_mo = uhf.eri_mo
fock_mo = uhf.fock_mo
chempot = uhf.chempot

# Build the auxiliaries via the MO energies and MO integrals for each spin:
se_ump2_alph = aux.build_ump2(e_mo, eri_mo[0], chempot=chempot)
se_ump2_beta = aux.build_ump2(e_mo[::-1], eri_mo[1][::-1], chempot=chempot[::-1])

# We can also use the arbitrary (w.r.t U/R) aux.build_mp2 function which detects
# the structure of the input arrays to get both spins at the same time:
se_ump2_alph, se_ump2_beta = aux.build_mp2(e_mo, eri_mo, chempot=chempot)

# Calculate the MP2 energies and compare them to canonical MP2:
e_mp2_a = 0.5 * (aux.energy_mp2(uhf.e[0], se_ump2_alph) + aux.energy_mp2(uhf.e[1], se_ump2_beta))
e_mp2_b = mp.MP2(uhf).run().e_corr
print('a) E(mp2) = %.12f' % e_mp2_a)
print('b) E(mp2) = %.12f' % e_mp2_b)


print('time elapsed: %d min %.4f s' % (timer.total() // 60, timer.total() % 60))
