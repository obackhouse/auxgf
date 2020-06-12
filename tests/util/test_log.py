import unittest
import numpy as np
from io import StringIO
import sys

from auxgf import mol
from auxgf.util import log


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        self.m = mol.Molecule(atoms='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz')

    @classmethod
    def tearDownClass(self):
        del self.m

    def test_warn(self):
        with self.assertWarns(Warning):
            log.warn('This is a test')

    def test_title(self):
        sys.stdout = stdout = StringIO()
        log.title('Test')
        sys.stdout = sys.__stdout__

    def test_iteration(self):
        sys.stdout = stdout = StringIO()
        log.iteration(0)
        sys.stdout = sys.__stdout__

    def test_molecule(self):
        sys.stdout = stdout = StringIO()
        log.molecule(self.m)
        sys.stdout = sys.__stdout__

    def test_options(self):
        sys.stdout = stdout = StringIO()
        log.options({'test': 'also a test'})
        sys.stdout = sys.__stdout__

    def test_timings(self):
        sys.stdout = stdout = StringIO()
        log.timings({'setup': 0.0})
        sys.stdout = sys.__stdout__

    def test_array(self):
        sys.stdout = stdout = StringIO()
        log.array(np.zeros((3, 3)), verbose=2)
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    unittest.main()
