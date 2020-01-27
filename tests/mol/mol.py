from auxgf import mol


mol1 = mol.Molecule(atoms='H 0 0 0; H 0 0 1', basis='6-31g', charge=-2, spin=2)

assert mol1.atoms == 'H 0 0 0; H 0 0 1'
assert mol1.basis == '6-31g'
assert mol1.charge == -2
assert mol1.spin == 2
assert mol1.natom == 2
assert mol1.nao == 4
assert mol1.nalph == 3
assert mol1.nbeta == 1
assert mol1.nelec == 4


mol2 = mol.Molecule.from_pyscf(mol1._pyscf)

assert mol2.atoms == 'H 0 0 0; H 0 0 1'
assert mol2.basis == '6-31g'
assert mol2.charge == -2
assert mol2.spin == 2
assert mol2.natom == 2
assert mol2.nao == 4
assert mol2.nalph == 3
assert mol2.nbeta == 1
assert mol2.nelec == 4
