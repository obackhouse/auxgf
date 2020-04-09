# Calculates IP/EA at RAGF2 level and compares to EOM-CCSD

import numpy as np
from pyscf.lib.linalg_helper import davidson
from auxgf import mol, hf, cc, aux, agf2
from auxgf.util import Timer

timer = Timer()


# Build the Molecule object:
m = mol.Molecule(atoms='H 0 0 0; Li 0 0 1.64', basis='cc-pvdz')

# Build the RHF object:
rhf = hf.RHF(m)
rhf.run()

# Run some RAGF2 calculations:
gf2_a = agf2.RAGF2(rhf, nmom=(1,1), verbose=False).run() 
gf2_b = agf2.RAGF2(rhf, nmom=(2,2), verbose=False).run()
gf2_c = agf2.RAGF2(rhf, nmom=(3,3), verbose=False).run()

# Method 1: using very little aux.Aux functionality
f_ext = np.block(([[rhf.get_fock(gf2_a.rdm1, basis='mo'), gf2_a.se.v],
                   [gf2_a.se.v.T,                         np.diag(gf2_a.se.e)]]))
e, c = np.linalg.eigh(f_ext)
c = c[:rhf.nao]
occ, vir = e < gf2_a.chempot, e >= gf2_a.chempot
e_hoqmo, e_luqmo = e[occ][-1], e[vir][0]  # only works because the eigenvalues are sorted
v_hoqmo, v_luqmo = c[:,occ][:,-1], c[:,vir][:,0]
print('RAGF2(1,1):')
print('IP = %.6f  weight = %.6f' % (-e_hoqmo, np.linalg.norm(v_hoqmo)))
print('EA = %.6f  weight = %.6f' % (e_luqmo, np.linalg.norm(v_luqmo)))
print('Gap = %.6f' % (e_luqmo - e_hoqmo))

# Method 2: using aux.Aux more:
e, c = gf2_b.se.eig(rhf.get_fock(gf2_b.rdm1, basis='mo'))
c = c[:rhf.nao]
gf = aux.Aux(e, c[:rhf.nao], chempot=gf2_b.chempot)
e_hoqmo, e_luqmo = gf.as_occupied().e[-1], gf.as_virtual().e[0]
v_hoqmo, v_luqmo = gf.as_occupied().v[:,-1], gf.as_virtual().v[:,0]
print('RAGF2(2,2):')
print('IP = %.6f  weight = %.6f' % (-e_hoqmo, np.linalg.norm(v_hoqmo)))
print('EA = %.6f  weight = %.6f' % (e_luqmo, np.linalg.norm(v_luqmo)))
print('Gap = %.6f' % (e_luqmo - e_hoqmo))

# Method 3: using a more efficient solver:
f_phys = rhf.get_fock(gf2_c.rdm1, basis='mo')
nroots = 5
def aop(x): return gf2_c.se.dot(f_phys, x)
def precond(dx, e, x0): return dx / (np.concatenate([np.diag(f_phys), gf2_c.se.e]) - e)
def pick(w, v, nroots, callback): 
    mask = np.argsort(abs(w-gf2_c.chempot))
    return w[mask], v[:,mask], 0
e, v = davidson(aop, np.eye(nroots, gf2_c.nphys+gf2_c.naux), precond, nroots=nroots, max_cycle=10*nroots, pick=pick)
e_hoqmo, e_luqmo = np.max(e[e < gf2_c.chempot]), np.min(e[e >= gf2_c.chempot])
v_hoqmo, v_luqmo = v[np.argmax(e[e < gf2_c.chempot])], v[np.argmin(e[e >= gf2_c.chempot])]
print('RAGF2(3,3):')
print('IP = %.6f  weight = %.6f' % (-e_hoqmo, np.linalg.norm(v_hoqmo[:rhf.nao])))
print('EA = %.6f  weight = %.6f' % (e_luqmo, np.linalg.norm(v_luqmo[:rhf.nao])))
print('Gap = %.6f' % (e_luqmo - e_hoqmo))

# CCSD:
ccsd = cc.CCSD(rhf)
ccsd.run()
e_ip, v_ip = ccsd._pyscf.ipccsd()[:2]
e_ea, v_ea = ccsd._pyscf.eaccsd()[:2]
print('EOM-CCSD:')
print('IP = %.6f  weight = %.6f' % (e_ip, np.linalg.norm(v_ip[:rhf.nao])))
print('EA = %.6f  weight = %.6f' % (e_ea, np.linalg.norm(v_ea[:rhf.nao])))
print('Gap = %.6f' % (e_ip + e_ea))


print('time elapsed: %d min %.4f s' % (timer.total() // 60, timer.total() % 60))
