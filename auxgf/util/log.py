''' Output stuff.
'''

import numpy as np
import sys
import warnings
from collections import OrderedDict


def warn(msg):
    warnings.warn('WARNING: %s' % msg)


def write(s, verbose=1):
    if verbose:
        sys.stdout.write(s)
        sys.stdout.flush()


def title(s, verbose=1):
    if verbose:
        n = len(s) + 2
        string = '\n'
        string += (' ' + '-' * n + ' ').center(32)
        string += '\n'
        string += ('  ' + s + '  ').center(32)
        string += '\n'
        string += (' ' + '-' * n + ' ').center(32)
        string += '\n\n'
        write(string)


def iteration(n, verbose=1):
    if verbose:
        string = '\n'
        string += ' --------------'.center(32) + '\n'
        string += (' Iteration %2d' % n).center(32) + '\n'
        string += ' --------------'.center(32) + '\n\n'
        write(string)


def molecule(mol, verbose=1):
    if verbose:
        labels = mol.labels
        coords = mol.coords
        
        s  = '-'*47 + '\n'
        s += ' %6s %12s %12s %12s \n' % ('label', 'x', 'y', 'z')
        s += '-'*47 + '\n'

        for i in range(mol.natom):
            tup = (labels[i], coords[i][0], coords[i][1], coords[i][2])
            s += ' %6s %12.6f %12.6f %12.6f\n' % tup

        s += '-'*47 + '\n'

        write(s)


def options(opts, verbose=1):
    max_size = max([len(x) for x in opts.keys()])
    if verbose:
        for item in sorted(opts.items()):
            if item[0][0] != '_':
                if isinstance(item[1], np.ndarray):
                    item = (item[0], item[1].ravel())
                write(('%-' + str(max_size) + 's : %-16s\n') % item)


def timings(times, verbose=1):
    if verbose:
        ref = OrderedDict()
        ref['setup']  = 'Setup'
        ref['build']  = 'Build'
        ref['gf']     = 'GF'
        ref['se']     = 'SE'
        ref['fock']   = 'Fock'
        ref['rpa']    = 'RPA'
        ref['merge']  = 'Merge'
        ref['fit']    = 'Fit'
        ref['energy'] = 'Energy'
        ref['total']  = 'Total'

        for key,val in ref.items():
            if key in times.keys():
                write('%-10s : %-10.3f\n' % (val, times[key]))


def array(arr, title, verbose=1):
    assert arr.ndim <= 2

    if arr.ndim == 1:
        arr = arr[None,:]

    if verbose > 1:
        with np.printoptions(precision=12, edgeitems=100, linewidth=200):
            line = '-' * (13 * arr.shape[1] + 1)

            s = '%s:\n' % title
            s += line + '\n'

            for i in range(arr.shape[0]):
                s += ' ' + ' '.join(['%12.6f' % x for x in arr[i,:]]) + ' \n'

            s += line + '\n'

            write(s)

