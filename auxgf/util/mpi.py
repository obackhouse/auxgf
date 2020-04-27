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









