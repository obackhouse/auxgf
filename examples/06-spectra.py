# Builds some spectra and self-energies using the aux.Aux functionality

import numpy as np
from auxgf import mol, hf, aux, agf2, grids
from auxgf.util import Timer

timer = Timer()


# Build the Molecule object:
m = mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='cc-pvdz')

# Build the RHF object:
rhf = hf.RHF(m)
rhf.run()

# Build the grid:
refq = grids.ReFqGrid(2**8, minpt=-5, maxpt=5, eta=0.1)

# Build the Hartree-Fock Green's function:
g_hf = aux.Aux(np.zeros(0), np.zeros((rhf.nao, 0)), chempot=rhf.chempot)

# Build the MP2 self-energy:
s_mp2 = aux.build_rmp2_iter(g_hf, rhf.fock_mo, rhf.eri_mo)

# Build the second-iteration Green's function, which corresponds to the QP spectrum at MP2 level or G^(2):
e, c = s_mp2.eig(rhf.fock_mo)
g_2 = g_hf.new(e, c[:rhf.nao])  # inherits g_hf.chempot

# Run an RAGF2 calcuation and get the converged Green's function and self-energy (we also use the RAGF2 density):
gf2 = agf2.RAGF2(rhf, nmom=(2,3), verbose=False)
gf2.run()
s_gf2 = gf2.se
e, c = s_gf2.eig(rhf.get_fock(gf2.rdm1, basis='mo'))
g_gf2 = s_gf2.new(e, c[:rhf.nao])

# For each Green's function, get the spectrum (Aux.as_spectrum only represents the function on a grid, we must
# also provide ordering='retarded' and then refactor):
def aux_to_spectrum(g):
    a = g.as_spectrum(refq, ordering='retarded')
    a = a.imag / np.pi
    a = a.trace(axis1=1, axis2=2)
    return a

a_hf = aux_to_spectrum(g_hf)
a_2 = aux_to_spectrum(g_2)
a_gf2 = aux_to_spectrum(g_gf2)

# Compare the spectra quantitatively:
print('| A(hf) - A(2) |   = %.12f' % (np.linalg.norm(a_hf - a_2) / refq.npts))
print('| A(gf2) - A(2) |  = %.12f' % (np.linalg.norm(a_gf2 - a_2) / refq.npts))
print('| A(gf2) - A(hf) | = %.12f' % (np.linalg.norm(a_gf2 - a_hf) / refq.npts))


print('time elapsed: %d min %.4f s' % (timer.total() // 60, timer.total() % 60))
