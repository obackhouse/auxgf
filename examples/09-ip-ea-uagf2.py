# Calculates IP/EA at UAGF2 level and compares to EOM-CCSD

import numpy as np
from pyscf.lib.linalg_helper import davidson
from auxgf import mol, hf, cc, aux, agf2
from auxgf.util import Timer

timer = Timer()


# Build the Molecule object:
m = mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='cc-pvdz')

# Build the UHF object:
uhf = hf.UHF(m)
uhf.run()

# Run some UAGF2 calculations:
gf2_a = agf2.UAGF2(uhf, nmom=(1,1), verbose=False).run() 
gf2_b = agf2.UAGF2(uhf, nmom=(2,2), verbose=False).run()
gf2_c = agf2.UAGF2(uhf, nmom=(3,3), verbose=False).run()

# Method 2: using aux.Aux more:
fock = uhf.get_fock(gf2_b.rdm1, basis='mo')
ea, ca = gf2_b.se[0].eig(fock[0])
eb, cb = gf2_b.se[1].eig(fock[1])
ca = ca[:uhf.nao]
cb = cb[:uhf.nao]
gfa = aux.Aux(ea, ca[:uhf.nao], chempot=gf2_b.chempot[0])
gfb = aux.Aux(eb, cb[:uhf.nao], chempot=gf2_b.chempot[1])
ea_hoqmo, ea_luqmo = gfa.as_occupied().e[-1], gfa.as_virtual().e[0]
eb_hoqmo, eb_luqmo = gfb.as_occupied().e[-1], gfb.as_virtual().e[0]
va_hoqmo, va_luqmo = gfa.as_occupied().v[:,-1], gfa.as_virtual().v[:,0]
vb_hoqmo, vb_luqmo = gfb.as_occupied().v[:,-1], gfb.as_virtual().v[:,0]
e_hoqmo, v_hoqmo = (ea_hoqmo, va_hoqmo) if ea_hoqmo < eb_hoqmo else (eb_hoqmo, vb_hoqmo)
e_luqmo, v_luqmo = (ea_luqmo, va_luqmo) if ea_luqmo > eb_luqmo else (eb_luqmo, vb_luqmo)
print('UAGF2(2,2):')
print('IP = %.6f  weight = %.6f' % (-e_hoqmo, np.linalg.norm(v_hoqmo)))
print('EA = %.6f  weight = %.6f' % (e_luqmo, np.linalg.norm(v_luqmo)))
print('Gap = %.6f' % (e_luqmo - e_hoqmo))

# CCSD:
ccsd = cc.CCSD(uhf)
ccsd.run()
e_ip, v_ip = ccsd._pyscf.ipccsd()[:2]
e_ea, v_ea = ccsd._pyscf.eaccsd()[:2]
print('EOM-CCSD:')
print('IP = %.6f  weight = %.6f' % (e_ip, np.linalg.norm(v_ip[:uhf.nao])))
print('EA = %.6f  weight = %.6f' % (e_ea, np.linalg.norm(v_ea[:uhf.nao])))
print('Gap = %.6f' % (e_ip + e_ea))


print('time elapsed: %d min %.4f s' % (timer.total() // 60, timer.total() % 60))
