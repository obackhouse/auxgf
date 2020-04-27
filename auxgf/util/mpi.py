''' Functions to run auxgf in parallel.
'''

try:
    from mpi4py import MPI as mpi
except ImportError:
    mpi = None

from auxgf.util import log

if mpi is not None:
    comm = mpi.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
else:
    comm = None
    size = 1
    rank = 0


def reduce_and_broadcast(m, op='sum', root=0):
    ''' Reduce a matrix onto the root process and then broadcast it
        to all other processes.
    '''

    if size == 1:
        return m

    if op == 'sum': 
        op = mpi.SUM
    elif op == 'max': 
        op = mpi.MAX
    elif op == 'min': 
        op = mpi.MIN
    elif op == 'sum': 
        op = mpi.SUM
    elif op == 'prod': 
        op = mpi.PROD
    elif op == 'land':
        op = mpi.LAND
    elif op == 'lor': 
        op = mpi.LOR
    elif op == 'band': 
        op = mpi.BAND
    elif op == 'bor': 
        op = mpi.BOR
    elif op == 'maxloc': 
        op = mpi.MAXLOC
    elif op == 'minloc': 
        op = mpi.MINLOC
    else:
        raise ValueError

    m_red = np.zeros_like(m)
    comm.Reduce(m, m_red, op=op, root=root)

    m = m_red
    comm.Bcast(m, root=0)

    return m


def wait_all():
    ''' Forces the program to wait for all processes to catch up.
    '''
    
    for i in range(size):
        comm.send(1, dest=i)

    for i in range(size):
        comm.recv(source=i)


def tril_indices_rows(nrows):
    ''' Fairly distributes (roughly) the row index of a lower-
        triangular matrix.

        i.e. nrows=10, size=4: [(0,9,4), (1,8,5), (2,7), (3,6)]

        This isn't exact but for nrows >> size it should be fine
    '''

    key = [[] for x in range(size)]
    top = False

    i = 0
    while True:
        for n in range(size):
            key[n].append(top)
            i += 1
            if i == nrows:
                break

        top = not top
        if i == nrows:
            break

    indices = list(range(nrows))
    indices_out = [[] for x in range(size)]

    while True:
        for n in range(size):
            if key[n].pop(0):
                indices_out[n].append(indices.pop(-1))
            else:
                indices_out[n].append(indices.pop(0))

            if not len(indices):
                break

        if not len(indices):
            break

    indices_out = [tuple(x) for x in indices_out]

    return indices_out

