import numpy as np

from auxgf import hf, mol, adc


mol1 = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto3g')

rhf1 = hf.RHF(mol1).run() 
uhf1 = hf.UHF(mol1).run()

adc3r1 = adc.ADC3(rhf1).run()
adc3u1 = adc.ADC3(uhf1).run()

assert np.allclose(adc3r1.e_tot, [adc3u1.e_tot])
