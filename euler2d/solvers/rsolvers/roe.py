from nbutils import array
from solvers.fluid import to_flux

import numba as nb
import numpy as np


def make_roe(gamma=1.4, nvars=4):
    """
    Roe's FDS 계산 함수 생성
    
    Parameters
    ----------
    gamma : float
        Specific heat ratio (기본값: 1.4)
    nvars : integer
        벡터 크기 (기본값: 4)
        
    Return
    ------
    _flux : function
        Roe flux 함수
    """
    ndims = nvars - 2

    @nb.jit(nopython=True)
    def _flux(ul, ur, nf, fn):
        fl, fr = array((nvars,)), array((nvars,))
        vl, vr = array((ndims,)), array((ndims,))
        dv, va = array((ndims,)), array((ndims,))
        ev = array((3, nvars))

        pl, contravl = to_flux(ul, fl, nf)
        pr, contravr = to_flux(ur, fr, nf)

        # Specific enthalpy, contra velocity for left / right
        for jdx in range(ndims):
            vl[jdx] = ul[jdx+1] / ul[0]
            vr[jdx] = ur[jdx+1] / ur[0]

        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        # Difference between two state
        drho = ur[0] - ul[0]
        dp = pr - pl
        dcontrav = contravr - contravl
        for jdx in range(ndims):
            dv[jdx] = vr[jdx] - vl[jdx]

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
        inv_aasq = 1/aasq

        l1 = np.abs(contrava - aa)
        l2 = np.abs(contrava)
        l3 = np.abs(contrava + aa)

        alp1 = 0.5*(dp - ra*aa*dcontrav)*inv_aasq
        alp2 = drho - dp*inv_aasq
        alp3 = 0.5*(dp + ra*aa*dcontrav)*inv_aasq
        
        ev[0, 0] = alp1
        ev[0, nvars-1] = alp1*(ha - aa*contrava)

        ev[1, 0] = alp2
        ev[1, nvars-1] = alp2*qq + ra*(va[0]*dv[0] + va[1]*dv[1] - contrava*dcontrav)

        ev[2, 0] = alp3
        ev[2, nvars-1] = alp3*(ha + aa*contrava)

        for jdx in range(ndims):
            ev[0, 1+jdx] = alp1*(va[jdx] - aa*nf[jdx])
            ev[1, 1+jdx] = alp2*va[jdx] + ra*(dv[jdx] - nf[jdx]*dcontrav)
            ev[2, 1+jdx] = alp3*(va[jdx] + aa*nf[jdx])

        for jdx in range(nvars):
            fn[jdx] = 0.5*(fl[jdx] + fr[jdx]) - 0.5*(l1*ev[0, jdx] + l2*ev[1, jdx] + l3*ev[2, jdx])

    return _flux
