import numpy as np

from auxgf import mol, hf, agf2

rhf = hf.RHF(mol.Molecule(atoms='H 0 0 0; H 0 0 0.75', basis='sto3g'))

nmom = (2,3)

gf2_a = agf2.RAGF2(rhf, nmom=nmom, damping=0.1, verbose=False)
gf2_a.run()

gf2_b = agf2.RAGF2(rhf, nmom=(None, nmom[-1]), damping=0.1, verbose=False)
gf2_b.run()

assert np.allclose(-1.12929514299, [gf2_a.e_tot, gf2_b.e_tot])
