# Runs some UAGF2 calculations

from auxgf import mol, hf, agf2
from auxgf.util import Timer

timer = Timer()


# Build the Molecule object:
m = mol.Molecule(atoms='H 0 0 0; Be 0 0 1', basis='cc-pvdz', spin=1)

# Build the UHF object:
uhf = hf.UHF(m)
uhf.run()

# Build the AGF2 object and run it for a few different settings:
# Simple UAGF2(1,1) setup without damping:
opts = dict(verbose=False, nmom=(1,1), damping=0.0)
gf2 = agf2.UAGF2(uhf, **opts)
gf2.run()
print('UAGF2(1,1): converged = %s  iterations = %d  E(corr) = %.12f' % (gf2.converged, gf2.iteration, gf2.e_corr))

# Tighten the Fock loop and increase to UAGF2(2,2):
opts.update(dict(nmom=(2,2), dtol=1e-10, diis_space=10, fock_maxiter=100, fock_maxruns=25))
gf2 = agf2.UAGF2(uhf, **opts)
gf2.run()
print('UAGF2(2,2): converged = %s  iterations = %d  E(corr) = %.12f' % (gf2.converged, gf2.iteration, gf2.e_corr))

# Tighten the AGF2 loop and add some damping:
opts.update(dict(etol=1e-7, maxiter=100, damping=0.25))
gf2 = agf2.UAGF2(uhf, **opts)
gf2.run()
print('UAGF2(2,2): converged = %s  iterations = %d  E(corr) = %.12f' % (gf2.converged, gf2.iteration, gf2.e_corr))

# Increase to UAGF2(3,3) and using the previous density matrix as a guess:
opts.update(dict(nmom=(3,3), dm0=gf2.rdm1))
gf2 = agf2.UAGF2(uhf, **opts)
gf2.run()
print('UAGF2(3,3): converged = %s  iterations = %d  E(corr) = %.12f' % (gf2.converged, gf2.iteration, gf2.e_corr))


print('time elapsed: %d min %.4f s' % (timer.total() // 60, timer.total() % 60))
