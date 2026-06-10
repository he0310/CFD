from nbutils import array
from solvers.fluid import to_primitive

import numba as nb
import numpy as np


def make_timestep(nfx, nfy, jac, si, sj, cfl, gamma=1.4, nvars=4, npads=(1.1)):
    """
    Local Time step calulcation kernel generator
    
    Parameters
    ----------
    nfx - integer
        xi 방향 격자 개수
    nfy - integer
        eta 방향 격자 개수
    jac : array
        Jacobian 크기
    si : array
        xi 방향 Face metric
    sj : array
        eta 방향 Face metric
    cfl : float
        CFL 값
    gamma : float
        비열비 (기본값 1.4)
    nvars: integer
        벡터 크기
    npad : integer
        Solution 벡터의 Padding 개수
        
    Returns
    -------
    _timestep : kernal
        시간 간격 계산 Kerenl
    """
    # Metric at the cell
    sic = 0.5*(si[:, :, :-1] + si[:, :, 1:])*jac
    sjc = 0.5*(sj[:, :-1, :] + sj[:, 1:, :])*jac

    # Magnitude of the metric
    aic = np.linalg.norm(sic, axis=0)
    ajc = np.linalg.norm(sjc, axis=0)
        
    @nb.jit(nopython=True)
    def _timestep(q, dt):
        # Allocate local static array
        qp = array((nvars,))
        
        dtmin = 1e8
        
        for j in range(nfy):
            jy = j + npads[1]
            for i in range(nfx):
                ix = i + npads[0]
                
                to_primitive(q[:, jy, ix], qp)
                rho, u, v, p = qp
                a = np.sqrt(gamma*p/rho)

                ui = sic[0, j, i]*u + sic[1, j, i]*v
                vj = sjc[0, j, i]*u + sjc[1, j, i]*v

                dt[jy, ix]  = cfl  / ((abs(ui) + a*aic[j, i]) + (abs(vj) + a*ajc[j, i]))
    
    return _timestep
