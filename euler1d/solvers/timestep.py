import numpy as np
from solvers.flux import to_primitive

def make_timestep(nfx, dx, cfl, gamma=1.4, nvars=3, npad=1):
    """
    Time step calulcation kernel generator
    
    Parameters
    ----------
    nfx : integer
        격자 (Number of Cell) 개수
    dx : float
        격자 간격
    cfl : float
        CFL 값
    gamma : float
        비열비 (기본값 1.4)
    nvars: integer
        변수 갯수 (기본값 3)
    npad : integer
        Solution 벡터의 Padding 개수
        
    Returns
    -------
    _timestep : kernal
        시간 간격 계산 Kerenl
    """
    def _timestep(u):
        qp = np.empty(nvars)
        dtmin = 1e8
        
        for i in range(npad, nfx+npad):
            # 원시변수로 변환
            to_primitive(u[:, i], qp)
            rho, v, p = qp
            # rho = np.maximum(rho, 1e-12)
            # p   = np.maximum(p, 1e-12)
            
            # 음속 계산
            a = np.sqrt(gamma*p/rho)
            
            # 최소 시간간격 계산
            dt = cfl * dx / (a + abs(v))
            dtmin = min(dt, dtmin)
            
        return dtmin
            
    return _timestep
