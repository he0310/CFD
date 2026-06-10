# Import implemented BC schemes!!!
from solvers.bcs.wall import make_bc_wall
from solvers.bcs.sup import make_bc_sup_in, make_bc_sup_out
from solvers.bcs.zero import make_bc_zero


def get_bc(name, nvars, **kwargs):
    bc = eval('make_bc_'+name)(nvars, **kwargs)
    return bc
