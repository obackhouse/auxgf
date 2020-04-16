# Runs some SCS-RAGF2 calculations

from auxgf import mol, hf, agf2
from auxgf.util import Timer

timer = Timer()


# Build the Molecule object:
m = mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='cc-pvdz')

# Build the RHF object:
rhf = hf.RHF(m)
rhf.run()

# Build the AGF2 object and run it with SCS scaling factors
opts = dict(verbose=False, nmom=(4,4), damping=0.0, os_factor=6/5, ss_factor=1/3)
gf2 = agf2.RAGF2(rhf, **opts)
gf2.run()
print('SCS-RAGF2(4,4): converged = %s  iterations = %d  E(corr) = %.12f' % (gf2.converged, gf2.iteration, gf2.e_corr))


print('time elapsed: %d min %.4f s' % (timer.total() // 60, timer.total() % 60))
