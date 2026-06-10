import numba as nb
import numpy as np

# Implement forward & Backward sweeps
@nb.jit(nopython=True)
def to_flux_pm_jac(q, j, n, d, kappa=1.01, gamma=1.4):
    """
    Compute signed flux jacobian of A+, A-

    Parameters
    ----------
    q : array
        conservative variables
        [rho, rho*u, rho*v, rho*et]
    j : array
        Jacobian
    n : array
        unit normal vector
    d : float
        direction (+1.0, or -1.0)
    kappa : float
        margin
    gamma : float
        비열비
    """
    u = q[1] / q[0]
    v = q[2] / q[0]
    et = q[3] / q[0]
    qq = 0.5*(u**2 + v**2)
    
    p = (gamma -1)*(q[3] - q[0]*qq)
    c = np.sqrt(gamma*p/q[0])
    ht = et + p/q[0]
    
    uu = n[0]*u + n[1]*v    
    qqb = (gamma-1)*qq
    
    # cell 법선 방향의 Flux jacobian
    
    # mass
    j[0,0] = 0
    j[0,1] = n[0]
    j[0,2] = n[1]
    j[0,3] = 0
    
    # x momentum
    j[1,0] = n[0]*qqb - uu*u
    j[1,1] = uu + n[0]*u*(2 - gamma)
    j[1,2] = n[1]*u -n[0]*v*(gamma-1)
    j[1,3] = n[0]*(gamma-1)
    
    # y momentum
    j[2,0] = n[1]*qqb - uu*v
    j[2,1] = n[0]*v - n[1]*u*(gamma-1)
    j[2,2] = uu + n[1]*v*(2 - gamma)
    j[2,3] = n[1]*(gamma-1)
    
    # energy
    j[3,0] = uu*(qqb - ht)
    j[3,1] = n[0]*ht - uu*u*(gamma-1)
    j[3,2] = n[1]*ht - uu*v*(gamma-1)
    j[3,3] = uu*gamma
    
    # Spectral radius, 보정계수
    r = kappa*(abs(uu) + c)
    
    # Positive or Negative
    j *= 0.5
    j[0,0] += d*0.5*r
    j[1,1] += d*0.5*r
    j[2,2] += d*0.5*r
    j[3,3] += d*0.5*r    
