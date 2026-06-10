from nbutils import array
from solvers.fluid import to_flux

import numba as nb
import numpy as np


def make_hllc(gamma=1.4, nvars=3):
    """
    HLLC flux 계산 함수 생성
    
    Parameters
    ----------
    gamma : float
        Specific heat ratio (기본값: 1.4)
    nvars : integer
        벡터 크기 (기본값: 3)
        
    Return
    ------
    _flux : function
        HLLC FDS flux 함수
    """
    ndims = nvars - 2
    
    @nb.jit(nopython=True)
    def _flux(ul, ur, nf, fn):
        # Allocate local static array
        fl, fr = array((nvars,)), array((nvars,))
        usl, usr = array((nvars,)), array((nvars,))
        vl, vr, va = array((ndims,)), array((ndims,)), array((ndims,))

        pl, contravl = to_flux(ul, fl, nf)
        pr, contravr = to_flux(ur, fr, nf)

        # Specific enthalpy, contra velocity for left / right
        for jdx in range(ndims):
            vl[jdx] = ul[jdx+1] / ul[0]
            vr[jdx] = ur[jdx+1] / ur[0]

        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]
        
        # Speed of sound
        al = np.sqrt(gamma*pl/ul[0])
        ar = np.sqrt(gamma*pr/ur[0])
        
        # Compute Roe averaged density and enthalpy
        rrr = np.sqrt(ur[0]/ul[0])
        ratl = 1.0/(1.0 + rrr)
        ratr = rrr*ratl
        ra = rrr*ul[0]
        ha = hl*ratl + hr*ratr

        for jdx in range(ndims):
            va[jdx] = vl[jdx]*ratl + vr[jdx]*ratr

        contrava = va[0]*nf[0] + va[1]*nf[1]

        qq = 0.5*(va[0]**2 + va[1]**2)
        aasq = (gamma - 1)*(ha - qq)
        aa = np.sqrt(aasq)
                
        # Estimate the left and right wave spees, sl and sr
        sl = min(contrava - aa, contravl - al)
        sr = max(contrava + aa, contravr + ar)
        ss = (pr - pl + ul[0]*contravl*(sl - contravl) - ur[0]*contravr*(sr - contravr)) \
            / (ul[0]*(sl - contravl) - ur[0]*(sr - contravr))
        
        # Intermediat states
        ul_com = (sl - contravl)/(sl - ss)
        usl[0] = ul[0]*ul_com
        usl[1] = usl[0]*(vl[0] + (ss - contravl)*nf[0])
        usl[2] = usl[0]*(vl[1] + (ss - contravl)*nf[1])
        usl[3] = ul_com*(ul[nvars-1] + (ss - contravl)*(ul[0]*ss + pl/(sl - contravl)))

        ur_com = (sr - contravr)/(sr - ss)
        usr[0] = ur[0]*ur_com
        usr[1] = usr[0]*(vr[0] + (ss - contravr)*nf[0])
        usr[2] = usr[0]*(vr[1] + (ss - contravr)*nf[1])
        usr[3] = ur_com*(ur[nvars-1] + (ss - contravr)*(ur[0]*ss + pr/(sr - contravr)))
        
        if 0 <= sl:
            for jdx in range(nvars):
                fn[jdx] = fl[jdx]
        elif 0 >= sr:
            for jdx in range(nvars):
                fn[jdx] = fr[jdx]
        elif 0 <= ss:
            for jdx in range(nvars):
                fn[jdx] = (fl[jdx] +sl*(usl[jdx] - ul[jdx]))
        else:
            for jdx in range(nvars):
                fn[jdx] = (fr[jdx] +sr*(usr[jdx] - ur[jdx]))
            
    return _flux
