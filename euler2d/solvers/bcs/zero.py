import numba as nb


def make_bc_zero(nvars=4, **kwargs):
    """
    Zero gradient BC function generator

    Parameter
    ---------
    nvars : integer
        변수 개수

    Return
    ------
    _bc : jitted function
        Zero BC function
    """
    @nb.jit(nopython=True)
    def _bc(ul, ur, nf):
        for k in range(nvars):
            ur[k] = ul[k]
            
    return _bc
