import numpy as np

def make_rusanov(gamma=1.4, nvars=3):
    """
    Rusanov flux 계산 함수 생성
    
    Parameters
    ----------
    gamma : float
        Specific heat ratio (기본값: 1.4)
    nvars : integer
        벡터 크기 (기본값: 3)
        
    Return
    ------
    _flux : function
        Rusanov flux 함수
    """
    fl, fr = np.empty(nvars), np.empty(nvars)
    
    def _flux(ul, ur, fn):
        pl, vl = to_flux(ul, fl)
        pr, vr = to_flux(ur, fr)
        
        vn = 0.5*(vl + vr)
        an = np.sqrt(gamma*(pl+pr) / (ul[0] + ur[0])) + np.abs(vn)
        
        for jdx in range(nvars):
            fn[jdx] = 0.5*(fl[jdx] + fr[jdx]) - 0.5*an*(ur[jdx] - ul[jdx])
            
    return _flux
