import numba as nb


def make_bc_wall(nvars=4, **kwargs):
    """
    Slip Wall BC function generator

    Parameter
    ---------
    nvars : integer
        변수 개수

    Return
    ------
    _bc : jitted function
        BC function
    """
    @nb.jit(nopython=True)
    def _bc(ul, ur, n):
        # Implement Wall BC!!!
        # After wall
        qn = 2*(ul[1]*n[0] + ul[2]*n[1])
        
        # Before wall (ghost cell)
        ur[1] = ul[1] - qn*n[0]
        ur[2] = ul[2] - qn*n[1]
        
        # At wall
        ur[0] = ul[0]
        ur[3] = ul[3]
        
    return _bc
