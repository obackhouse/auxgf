import numpy as np
import auxgf


def rand_1e(n, scale=1.0, hermi=True):
    ''' Builds a random 1-electron matrix.
    '''

    m = scale * (np.random.random((n, n)) - 0.5)

    if hermi:
        m = 0.5 * (m + m.T)

    return m


def rand_2e(n, scale=1.0, hermi=True):
    ''' Builds a random 2-electron matrix.
    '''

    m = scale * (np.random.random((n, n, n, n)) - 0.5)

    if hermi:
        m = 0.125 * (m + 
                     m.transpose(1,0,2,3) + 
                     m.transpose(0,1,3,2) +
                     m.transpose(1,0,3,2) +
                     m.transpose(2,3,0,1) + 
                     m.transpose(2,3,1,0) + 
                     m.transpose(3,2,0,1) +
                     m.transpose(3,2,1,0))

    return m


def build_rand_aux(nphys, naux, e_scale=1.0, v_scale=0.5):
    ''' Builds a set of random auxiliaries.
    '''

    e = e_scale * (np.random.random((naux)) - 0.5)
    v = v_scale * (np.random.random((nphys, naux)) - 0.5)

    return auxgf.aux.Aux(e, v)


def build_rmp2_aux(atoms, basis, charge=0, spin=0):
    ''' Builds a set of MP2 poles for a restricted reference directly
        from the molecular information.
    '''

    mol = auxgf.mol.Molecule(atoms=atoms, basis=basis, charge=charge, spin=spin)
    rhf = auxgf.hf.RHF(mol).run()
    se = auxgf.aux.build_mp2(rhf.e, rhf.eri_mo, rhf.chempot)
    return se


def build_ump2_aux(atoms, basis, charge=0, spin=0):
    ''' Builds a set of MP2 poles for an unrestricted reference 
        directly from the molecular information.
    '''

    mol = auxgf.mol.Molecule(atoms=atoms, basis=basis, charge=charge, spin=spin)
    uhf = auxgf.hf.UHF(mol).run()
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
