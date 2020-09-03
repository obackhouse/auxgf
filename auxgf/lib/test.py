import numpy as np
from auxgf import mol, hf, agf2

#m = mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='6-31g')
m = mol.Molecule(atoms='O 0 0 0; O 0 0 1', basis='aug-cc-pvdz')
rhf = hf.RHF(m, with_df=True).run()
ragf2 = agf2.OptRAGF2(rhf, verbose=False).run()
