import numpy as np


def make_tvdrk3(nfx, nfy, rhside, nvars=4, npads=(1,1)):
    """
    TVD Runge Kutta 3rd 계산 커널 생성
    
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
    u0 = np.zeros((nvars, nfy+2*npads[1], nfx+2*npads[0]))
    
    # Temporal arrays for flux
    ff = np.zeros((nvars, nfy, nfx+1))
    fg = np.zeros((nvars, nfy+1, nfx))
    
    def _step(dt, u):
        # TVD-RK3
        rhside(u, du, ff, fg)
        u0[:] = u
        u += dt*du

        rhside(u, du, ff, fg)
        u[:] = (3*u0 + u +dt*du)/4

        rhside(u, du, ff, fg)
        u[:] = (u0 + 2*u + 2*dt*du)/3
        
    return _step
