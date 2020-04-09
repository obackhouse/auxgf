import numpy as np
from auxgf import *
from auxgf import agwa

m = mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='cc-pvdz')
m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='6-31g')
rhf = hf.RHF(m)
gw = agwa.RAGWA(rhf, nmom=(1,1), scheme='GW', verbose=False)
gw.run()
e_rpa, v_rpa, xpy = gw.solve_casida()
w_rpa = np.linalg.norm(v_rpa[:rhf.nao], axis=0)
print(rhf.nao)
print(e_rpa.size)
print(np.sum(w_rpa > 0.01))
