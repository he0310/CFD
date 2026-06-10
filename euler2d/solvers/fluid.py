import numba as nb
import numpy as np


@nb.jit(nopython=True)
def to_primitive(q, qp, gamma=1.4):
    """
    Converting primitive variables from conservative variables
    
    Parameters
    ----------
    q : vector
        Conservative variables
    qp: vector
        Primitive variables
    gamma : float
        specific heat ratio (default: 1.4)
    """
    rho = q[0]
    u = q[1] / rho
    v = q[2] / rho
    rhoe = (q[3] - 0.5*(q[1]**2 + q[2]**2) /q[0])
    p = (gamma-1)*rhoe
    
    qp[0] = rho
    qp[1] = u
    qp[2] = v
    qp[3] = p
    

@nb.jit(nopython=True)
def to_conservative(qp, q, gamma=1.4):
    """
    Converting conservative variables from primitive variables
    
    Parameters
    ----------
    qp: vector
        Primitive variables
    q : vector
        Conservative variables
    gamma : float
        specific heat ratio (default: 1.4)
    """
    rho, u, v, p = qp
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    rhoe = p / (gamma-1)
    q[3] = rhoe + 0.5*rho*(u**2 + v**2)
    

@nb.jit(nopython=True)    
def to_flux(q, f, n, gamma=1.4):
    """
    Flux calculation
    
    Parameters
    ----------
    q : vector
        Conservative variables
    f : vector
        Flux vector
    n : vector
        Unit normal direction vector
    gamma : float
        specific heat ratio (default: 1.4)

    Return
    ------
    p : float
        Pressure
    un : float
        Contravariant velocity
    """
    rho = q[0]
    u = q[1] / q[0]
    v = q[2] / q[0]
    et = q[3] / q[0]
    
    p = (gamma-1)*rho*(et - 0.5*(u**2 + v**2))
    ht = et + p/rho
    
    # Contravariant velocity
    un = u*n[0] + v*n[1]
    
    f[0] = rho*un
    f[1] = rho*un*u + n[0]*p
    f[2] = rho*un*v + n[1]*p
    f[3] = rho*un*ht
    
    return p, un


@nb.jit(nopython=True)    
def to_pressure(q, n, gamma=1.4):
    """
    Compute pressure
    
    Parameters
    ----------
    q : vector
        Conservative variables
    gamma : float
        specific heat ratio (default: 1.4)

    Return
    ------
    p : float
        Pressure
    un : float
        Contravariant velocity
    """
    rho = q[0]
    u = q[1] / q[0]
    v = q[2] / q[0]
    et = q[3] / q[0]
    
    p = (gamma-1)*rho*(et - 0.5*(u**2 + v**2))
    ht = et + p/rho
    
    # Contravariant velocity
    un = u*n[0] + v*n[1]
    
    return p, un
