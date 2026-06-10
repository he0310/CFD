from nbutils import array
from solvers.fluid import to_flux

import numba as nb
import numpy as np


def make_rusanov(gamma=1.4, nvars=4):
    """
    Rusanov flux 계산 함수 생성
    
    Parameters
    ----------
    gamma : float
        Specific heat ratio (기본값: 1.4)
    nvars : integer
        벡터 크기 (기본값: 4)
        
    Return
    ------
    _flux : function
        Rusanov flux 함수
    """
    
    @nb.jit(nopython=True)
    def _flux(ul, ur, n, fn):
        fl, fr = array((nvars,)), array((nvars,))
        
        pl, vl = to_flux(ul, fl, n)
        pr, vr = to_flux(ur, fr, n)
        
        vn = 0.5*(vl + vr)
        an = np.sqrt(gamma*(pl+pr) / (ul[0] + ur[0]))*np.sqrt(n[0]**2 + n[1]**2) + np.abs(vn)
        
        for jdx in range(nvars):
            fn[jdx] = 0.5*(fl[jdx] + fr[jdx]) - 0.5*an*(ur[jdx] - ul[jdx])
            
    return _flux
