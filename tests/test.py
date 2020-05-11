from collections import OrderedDict
import warnings
import time
import sys
warnings.simplefilter('ignore', FutureWarning)

from auxgf.util import mpi


modules = OrderedDict([
    ('util', ['ao2mo', 'linalg']),
    ('mol', ['mol']),
    ('hf', ['rhf', 'uhf']),
    ('dft', ['rdft', 'udft']),
    ('mp', ['mp2', 'oomp2']),
    ('cc', ['ccsd']),
    ('adc', ['adc2', 'adc2x', 'adc3']),
    ('aux', ['aux', 'build', 'energy']),
    ('agf2', ['ragf2', 'uagf2']),
])

if mpi.size > 1:
    if mpi.rank == 0:
        sys.stdout.write('%6s %12s %6s %6s\n' % ('proc', 'module', 'time', 'status'))
        sys.stdout.write('%6s %12s %6s %6s\n' % ('-'*6, '-'*12, '-'*6, '-'*6))
else:
    sys.stdout.write('%12s %6s %6s\n' % ('module', 'time', 'status'))
    sys.stdout.write('%12s %6s %6s\n' % ('-'*12, '-'*6, '-'*6))

passed = True
init_time = time.time()

for module,files in modules.items():
    for f in files:
        start = time.time()

        if mpi.size > 1:
            sys.stdout.write('%6d %12s ' % (mpi.rank, module + '/' + f))
        else:
            sys.stdout.write('%12s ' % (module + '/' + f))

        try:
            exec('from tests.%s import %s' % (module, f))
            sys.stdout.write('%6.4f %6s\n' % (time.time() - start, 'pass'))
        except:
            sys.stdout.write('%6.4f %6s\n' % (time.time() - start, 'fail'))
            passed = False

sys.stdout.flush()

if mpi.size > 1:
    mpi.wait_all()

if not passed:
    if mpi.size == 1:
        sys.stdout.write('Some tests failed, run individual files to see errors\n')
    else:
        sys.stdout.write('Some tests failed on proc %d, run individual files to see errors\n' % mpi.rank)

sys.stdout.flush()

if mpi.size > 1:
    mpi.wait_all()

if mpi.rank == 0:
    sys.stdout.write('Runtime: %6.4f s\n' % (time.time() - init_time))

