import numpy as np

from auxgf import mol, hf, agf2

uhf = hf.UHF(mol.Molecule(atoms='H 0 0 0; H 0 0 0.75', basis='sto3g')).run()

nmom = (2,4)

gf2_a = agf2.UAGF2(uhf, nmom=nmom, damping=0.1, verbose=False).run()
gf2_b = agf2.UAGF2(uhf, nmom=(None, nmom[-1]), damping=0.1, verbose=False).run()

assert np.allclose(-1.12929514299, [gf2_a.e_tot, gf2_b.e_tot])


uhf = hf.UHF(mol.Molecule(atoms='H 0 0 0; Be 0 0 1', basis='sto3g', spin=1)).run()

nmom = (2,4)

gf2_a = agf2.UAGF2(uhf, nmom=nmom, damping=0.1, verbose=False).run()
gf2_b = agf2.UAGF2(uhf, nmom=(None, nmom[-1]), damping=0.1, verbose=False).run()

assert np.allclose(-14.86986264197264-0.0212930003512, [gf2_a.e_tot, gf2_b.e_tot])
