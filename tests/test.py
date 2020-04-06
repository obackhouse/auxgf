from collections import OrderedDict
import warnings
import time
import sys
warnings.simplefilter('ignore', FutureWarning)


modules = OrderedDict([
    ('util', ['ao2mo', 'linalg']),
    ('mol', ['mol']),
    ('hf', ['rhf', 'uhf']),
    ('dft', ['rdft', 'udft']),
    ('mp', ['mp2', 'oomp2']),
    ('cc', ['ccsd']),
    ('aux', ['aux', 'build', 'energy']),
    ('agf2', ['ragf2', 'uagf2']),
])

sys.stdout.write('%12s %6s %6s\n' % ('module', 'time', 'status'))
sys.stdout.write('%12s %6s %6s\n' % ('-'*12, '-'*6, '-'*6))

passed = True
init_time = time.time()

for module,files in modules.items():
    for f in files:
        start = time.time()
        sys.stdout.write('%12s ' % (module + '/' + f))
        try:
            exec('from tests.%s import %s' % (module, f))
            sys.stdout.write('%6.4f %6s\n' % (time.time() - start, 'pass'))
        except:
            sys.stdout.write('%6.4f %6s\n' % (time.time() - start, 'fail'))
            passed = False

if not passed:
    sys.stdout.write('Some tests failed, run individual files to see errors\n')

sys.stdout.write('Runtime: %6.4f s\n' % (time.time() - init_time))

