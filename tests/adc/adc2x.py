import numpy as np

from auxgf import hf, mol, adc


mol1 = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto3g')

rhf1 = hf.RHF(mol1).run() 
uhf1 = hf.UHF(mol1).run()

adc2r1 = adc.ADC2x(rhf1).run()
adc2u1 = adc.ADC2x(uhf1).run()

assert np.allclose(adc2r1.e_tot, [adc2u1.e_tot])
