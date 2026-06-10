import numpy as np


def make_expliciteuler(nfx, nfy, rhside, nvars=4, npads=(1,1)):
    """
    Euler Explicit 계산 커널 생성
    
    Parameters
    ----------
    nfx - integer
        xi 방향 격자 개수
    nfy - integer
        eta 방향 격자 개수
    rhside : kernel
        우변 계산 커널
    nvars : integer
        벡터 크기 (기본값: 3)
    npad : integer
        Solution 벡터의 Padding 개수
    """
    # Temporal arrays
    du = np.zeros((nvars, nfy+2*npads[1], nfx+2*npads[0]))
    
    # Temporal arrays for flux
    ff = np.zeros((nvars, nfy, nfx+1))
    fg = np.zeros((nvars, nfy+1, nfx))
    
    def _step(dt, u):
        rhside(u, du, ff, fg)
        du[:] = dt*du
        u += du
        
        # Density residual
        return np.linalg.norm(du[0])

    return _step
