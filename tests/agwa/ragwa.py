import numpy as np

from auxgf import mol, hf, agwa

rhf = hf.RHF(mol.Molecule(atoms='H 0 0 0; H 0 0 0.75', basis='sto3g'))

nmom = (2,3)

gw_a = agwa.RAGWA(rhf, nmom=nmom, damping=0.1, scheme='GW', verbose=False)
gw_a.run()

gw_b = agwa.RAGWA(rhf, nmom=nmom, damping=0.1, scheme='GW', verbose=False)
gw_b.run()

