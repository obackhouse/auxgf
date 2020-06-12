''' Molecule class
'''

from pyscf import gto


class Molecule:
    ''' Molecule class.

    Parameters
    ----------
    atoms : list of tuples
        atom labels and coordinates as tuples of (label, (x, y, z)) or
        any relevant pyscf format
    basis : str
        basis set (default 'sto-3g')
    charge : int
        net charge (default 0)
    spin : int
        spin number = 2S (default 0)

    Attributes
    ----------
    atoms : list of tuples
        atom labels and coordinates as input
    basis : str
        basis set
    charge : int
        net charge
    spin : int
        spin number = 2S
    natom : int
        number of atoms
    nao : int
        number of atomic orbitals
    nalph : int
        number of alpha electron
    nbeta : int
        number of beta electrons
    nelec : int
        total number of electrons
    ncore : int
        number of core electrons
    e_nuc : float
        nuclear repulsion energy
    labels : list of str
        atomic labels
    coords : list of lists of float
        atomic coordinates

    Methods
    -------
    intor(*args, **kwargs)
        interface to pyscf.gto.Mole.intor, see pyscf documentation
        for arguments
    '''

    def __init__(self, atoms, **kwargs):
        self._pyscf = gto.M(atom=atoms, verbose=0, **kwargs)


    def intor(self, *args, **kwargs):
        ''' Interface to mol.intor from pyscf.
        '''

        return self._pyscf.intor(*args, **kwargs)


    @property
    def ncore(self):
        ''' Gets the number of core orbitals.
        '''

        ncore = 0
        key = [(4, 0), (12, 2), (30, 10), (38, 18), (48, 28), (56, 36)]

        for charge in self.charges:
            for np, ne in key:
                if charge <= np:
                    ncore += ne
                    break
            else:
                raise ValueError

        return ncore

    @property
    def natom(self): 
        return self._pyscf.natm

    @property
    def nao(self): 
        return self._pyscf.nao

    @property
    def nalph(self):
        return self._pyscf.nelec[0]

    @property
    def nbeta(self):
        return self._pyscf.nelec[1]

    @property
    def nelec(self):
        return self._pyscf.nelectron

    @property
    def e_nuc(self):
        return self._pyscf.energy_nuc()

    @property
    def labels(self):
        return [self._pyscf.atom_symbol(i) for i in range(self.natom)]

    @property
    def coords(self):
        return [self._pyscf.atom_coord(i) for i in range(self.natom)]

    @property
    def atoms(self):
        return self._pyscf.atom

    @property
    def charge(self):
        return self._pyscf.charge

    @property
    def charges(self):
        return self._pyscf.atom_charges()

    @property
    def spin(self):
        return self._pyscf.spin

    @property
    def basis(self):
        return self._pyscf.basis


    @classmethod
    def from_pyscf(cls, mol):
        ''' Builds the Molecule object from a pyscf.gto.Mole object.

        Parameters
        ----------
        mol : pyscf.gto.Mole
            molecule

        Returns
        -------
        mol : Molecule
            molecule
        '''

        _mol = Molecule('H 0 0 0; H 0 0 1')
        _mol._pyscf = mol

        return _mol
