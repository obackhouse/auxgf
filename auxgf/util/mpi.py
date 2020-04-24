''' Functions to run auxgf in parallel.
'''

try:
    from mpi4py import MPI as mpi
except ImportError:
    mpi = None

if mpi is not None:
    comm = mpi.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    _ops = { 'sum':    mpi.SUM, 
             'max':    mpi.MAX,
             'min':    mpi.MIN,
             'sum':    mpi.SUM,
             'prod':   mpi.PROD,
             'land':   mpi.LAND,
             'lor':    mpi.LOR,
             'band':   mpi.BAND,
             'bor':    mpi.BOR,
             'maxloc': mpi.MAXLOC,
             'minloc': mpi.MINLOC }
else:
    comm = None
    size = 1
    rank = 0
    _ops = None


def reduce_and_broadcast(m, op='sum', root=0):
    ''' Reduce a matrix onto the root process and then broadcast it
        to all other processes.
    '''

    if size == 1:
        return m

    op = _ops[op]

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
