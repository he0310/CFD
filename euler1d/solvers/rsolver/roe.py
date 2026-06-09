from nbutils import array
from solvers.fluid import to_flux

import numba as nb
import numpy as np


def make_roe(gamma=1.4, nvars=3):
    """
    Roe flux 계산 함수 생성
    
    Parameters
    ----------
    gamma : float
        Specific heat ratio (기본값: 1.4)
    nvars : integer
        벡터 크기 (기본값: 3)
        
    Return
    ------
    _flux : function
        Roe's FDS flux 함수
    """
    @nb.jit(nopython=True)
    def _flux(ul, ur, fn):
        ### DO it yourself.
        fl, fr = array(nvars), array(nvars)
        ram, alp = array(nvars), array(nvars)
        ev0, ev1, ev2 = array(nvars), array(nvars), array(nvars)
        
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)
        
        hl = (ul[-1] + pl)/ul[0]
        hr = (ur[-1] + pr)/ur[0]

        # Difference between two state
        drho = ur[0] - ul[0]
        dv = vr - vl
        dp = pr - pl
        
        # Compute Roe averaged density and enthalpy
        rrr = np.sqrt(ur[0]/ul[0])
        ratl = 1.0/(1.0 + rrr)
        ratr = rrr*ratl
        ra = rrr*ul[0]
        va = vl*ratl + vr*ratr
        ha = hl*ratl + hr*ratr
        aasq = (gamma-1)*(ha - 0.5*va**2)
        aa = np.sqrt(aasq)
                
        # Eigen Structure
        ram[0] = abs(va)
        ram[1] = abs(va + aa)
        ram[2] = abs(va - aa)
        
        alp[0] = drho - dp/aasq
        alp[1] = 0.5*(dp + ra*aa*dv)/aasq
        alp[2] = 0.5*(dp - ra*aa*dv)/aasq
        
        ev0[0] = 1.0
        ev0[1] = va
        ev0[2] = 0.5*va**2
        
        ev1[0] = 1.0
        ev1[1] = va + aa
        ev1[2] = ha + aa*va
        
        ev2[0] = 1.0
        ev2[1] = va - aa
        ev2[2] = ha - aa*va
                
        for jdx in range(3):
            fn[jdx] = 0.5*(fl[jdx] + fr[jdx]) \
            - 0.5*(ram[0]*alp[0]*ev0[jdx] + ram[1]*alp[1]*ev1[jdx] + ram[2]*alp[2]*ev2[jdx])
            
    return _flux
