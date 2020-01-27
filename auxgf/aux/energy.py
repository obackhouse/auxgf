''' Functions for computing energies of sets of auxiliaries.
'''

import numpy as np

from auxgf import util


def energy_2body_aux(gf, se, both_sides=False):
    ''' Calculates the two-body contribution to the electronic energy
        using the auxiliary representation of the Green's function and
        self-energy, according to the Galitskii-Migdal formula.

    Parameters
    ----------
    gf : Aux
        auxiliary representation of Green's function
    se : Aux
        auxiliary representation of self-energy
    both_sides : bool, optional
        if True, calculate both halves of the functional and return
        the mean, default False

    Returns
    -------
    e2b : float
        two-body contribution to electronic energy
    '''

    #TODO in C

    if isinstance(se, (tuple, list)):
        if isinstance(gf, (tuple, list)):
            return sum([energy_2body_aux(gf[i], se[i], both_sides=both_sides)
                        for i in range(len(se))]) / len(se)
        else:
            return sum([energy_2body_aux(gf, se[i], both_sides=both_sides)
                        for i in range(len(se))]) / len(se)

    nphys = se.nphys

    e2b = 0.0

    for l in range(gf.nocc):
        vxl = gf.v[:nphys,l]
        vxk = se.v[:,se.nocc:]

        dlk = 1.0 / (gf.e[l] - se.e[se.nocc:])

        e2b += util.einsum('xk,yk,x,y,k->', vxk, vxk, vxl, vxl, dlk)

    if both_sides:
        for l in range(gf.nocc, gf.naux):
            vxl = gf.v[:nphys,l]
            vxk = se.v[:,:se.nocc]

            dlk = -1.0 / (gf.e[l] - se.e[:se.nocc])

            e2b += util.einsum('xk,yk,x,y,k->', vxk, vxk, vxl, vxl, dlk)
    else:
        e2b *= 2.0

    return np.asscalar(e2b)


def energy_mp2_aux(mo, se, both_sides=False):
    ''' Calculates the two-body contribution to the electronic energy
        using the MOs and the auxiliary representation of the
        self-energy according the the MP2 form of the Galitskii-Migdal
        formula.

    Parameters
    ----------
    mo : (n) ndarray
        MO energies
    se : Aux
        auxiliary representation of self-energy
    both_sides : bool, optional
        if True, calculate both halves of the functional and return
        the mean, default False

    Returns
    -------
    e2b : float
        two-body contribution to electronic energy
    '''

    if isinstance(se, (tuple, list)):
        if np.asarray(mo).ndim == 2:
            return sum([energy_mp2_aux(mo[i], se[i], both_sides=both_sides)
                        for i in range(len(se))]) / len(se)
        else:
            return sum([energy_mp2_aux(mo, se[i], both_sides=both_sides)
                        for i in range(len(se))]) / len(se)

    nphys = se.nphys

    occ = mo < se.chempot
    vir = mo >= se.chempot

    vxk = se.v_vir[occ]
    dxk = 1.0 / util.outer_sum([mo[occ], -se.e_vir])

    e2b = util.einsum('xk,xk->', vxk**2, dxk)

    if both_sides:
        vxk = se.v_occ[vir]
        dxk = -1.0 / util.outer_sum([mo[vir], -se.e_occ])

        e2b += util.einsum('xk,xk->', vxk**2, dxk)
        e2b *= 0.5

    return np.asscalar(e2b)



























