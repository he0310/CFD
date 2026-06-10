import numba as nb
import numpy as np


def make_bc_sup_in(nvars=4, rhof=1.0, uf=1.0, vf=0.0, pf=1.0, gamma=1.4):
    """
    Supersonic inlet function generator

    Parameter
    ---------
    nvars : integer
        변수 개수
    rhof : float
        자유류 밀도
    uf : float
        자유류 x 방향 속도
    vf : float
        자유류 y 방향 속도
    pf : float
        자유류 압력
    gamma : float
        비열비 (기본값 : 1.4)

    Return
    ------
    _bc : jitted function
        BC function
    """
    # Conservative variables for free-stream condition
    etf = pf/rhof/(gamma-1) + 0.5*(uf**2 + vf**2)
    qf = np.array([rhof, rhof*uf, rhof*vf, rhof*etf])

    @nb.jit(nopython=True)
    def _bc(ul, ur, nf):
        for k in range(nvars):
            ur[k] = qf[k]
            
    return _bc


def make_bc_sup_out(nvars=4, **kwargs):
    """
    Supersonic outlet function generator

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
        for k in range(nvars):
            ur[k] = ul[k]
            
    return _bc
