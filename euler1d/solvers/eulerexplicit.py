import numpy as np

def make_eulerexplicit(nfx, rhside, nvars=3, npad=1):
    """
    Euler Explicit 계산 커널 생성
    
    Parameters
    ----------
    nfx : integer
        격자 개수
    rhside : kernel
        우변 계산 커널
    nvars : integer
        벡터 크기 (기본값: 3)
    npad : integer
        Solution 벡터의 Padding 개수
    """
    # Temporal arrays
    f = np.zeros((nvars, nfx+1))
    du = np.zeros((nvars, nfx+2*npad))
    
    def _step(dt, u):
        rhside(u, du, f)
        u += dt*du

    return _step
