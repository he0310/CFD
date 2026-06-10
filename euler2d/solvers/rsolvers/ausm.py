from nbutils import array
from solvers.fluid import to_pressure

import numba as nb
import numpy as np


def make_ausmp(gamma=1.4, nvars=3):
    """
    AUSM+ flux 계산 함수 생성
    
    Parameters
    ----------
    gamma : float
        Specific heat ratio (기본값: 1.4)
    nvars : integer
        벡터 크기 (기본값: 3)
        
    Return
    ------
    _flux : function
        AUSM+ FVS 함수    
    """
    alpha = 3/16
    beta = 1/8
    ndims = nvars - 2

    @nb.jit(nopython=True)
    def _flux(ul, ur, nf, fn): 
        pl, vl = to_pressure(ul, nf)
        pr, vr = to_pressure(ur, nf)
        
        # Speed of sound
        al = np.sqrt(gamma*pl/ul[0])
        ar = np.sqrt(gamma*pr/ur[0])      
        
        # Enthalpy
        htl = (ul[3] + pl)/ul[0]
        htr = (ur[3] + pr)/ur[0]
        
        # Critical (or Sonic) speed of sound
        asl = np.sqrt(2*htl*(gamma - 1)/(gamma + 1))
        asr = np.sqrt(2*htr*(gamma - 1)/(gamma + 1))
        
        # tilde a
        atl = asl**2 / max(asl, abs(vl))
        atr = asr**2 / max(asr, abs(vr))
        
        # Common speed of sound
        amid = min(atl, atr)
        #amid = np.sqrt(al*ar)
        
        # Mach numbers
        Ml = vl / amid
        Mr = vr / amid
        
        # Mach splitting function
        if abs(Ml) < 1:
            Mp = 0.25*(Ml + 1.)**2          + beta*(Ml**2 - 1.)**2
            Pp = 0.25*(Ml + 1.)**2*(2 - Ml) + alpha*Ml*(Ml**2 -1.)**2
        else:
            Mp = 0.5*(Ml + abs(Ml))
            Pp = 0.5*(Ml + abs(Ml)) / Ml
            
        if abs(Mr) < 1:
            Mm = -0.25*(Mr - 1.)**2          - beta*(Mr**2 - 1.)**2
            Pm =  0.25*(Mr - 1.)**2*(2 + Mr) - alpha*Mr*(Mr**2 -1.)**2
        else:
            Mm = 0.5*(Mr - abs(Mr))
            Pm = 0.5*(Mr - abs(Mr)) / Mr
            
        Mmid = Mp + Mm
        pmid = Pp*pl + Pm*pr
        
        if Mmid > 0:
            fn[0] = amid*Mmid*ul[0]
            fn[1] = amid*Mmid*ul[1]
            fn[2] = amid*Mmid*ul[2]
            fn[3] = amid*Mmid*ul[0]*htl
        else:
            fn[0] = amid*Mmid*ur[0]
            fn[1] = amid*Mmid*ur[1]
            fn[2] = amid*Mmid*ur[2]
            fn[3] = amid*Mmid*ur[0]*htr
            
        fn[1] += pmid*nf[0]
        fn[2] += pmid*nf[1]
            
    return _flux
