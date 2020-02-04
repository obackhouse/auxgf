import numpy as np
import auxgf


def build_rmp2_aux(atoms, basis, charge=0, spin=0):
    ''' Builds a set of MP2 poles for a restricted reference directly
        from the molecular information.
    '''

    mol = auxgf.mol.Molecule(atoms=atoms, basis=basis, charge=charge, spin=spin)
    rhf = auxgf.hf.RHF(mol)
    se = auxgf.aux.build_mp2(rhf.e, rhf.eri_mo, rhf.chempot)
    return se


def build_ump2_aux(atoms, basis, charge=0, spin=0):
    ''' Builds a set of MP2 poles for an unrestricted reference 
        directly from the molecular information.
    '''

    mol = auxgf.mol.Molecule(atoms=atoms, basis=basis, charge=charge, spin=spin)
    uhf = auxgf.hf.UHF(mol)
    se = auxgf.aux.build_mp2(uhf.e, uhf.eri_mo, uhf.chempot)
    return se


def build_mp2_aux(atoms, basis, charge=0, spin=0):
    ''' Builds a set of MP2 poles directly from the molecular 
        information.
    '''

    if spin % 2 == 0:
        return build_rmp2_aux(atoms, basis, charge=charge, spin=spin)
    else:
        return build_ump2_aux(atoms, basis, charge=charge, spin=spin)
