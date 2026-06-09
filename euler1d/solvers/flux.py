import numpy as np

def to_primitive(u, up, gamma=1.4):
    """
    Converting primitive variables from conservative variables
    
    Parameters
    ----------
    u : vector
        Conservative variables
    up: vector
        Primitive variables
    gamma : float
        specific heat ratio (default: 1.4)
    """
    rho = u[0]
    v = u[1] / rho
    rhoe = (u[2] - 0.5*u[1]**2/u[0])
    p = (gamma-1)*rhoe
    
    up[0] = rho
    up[1] = v
    up[2] = p
    
    
def to_conservative(up, u, gamma=1.4):
    """
    Converting conservative variables from primitive variables
    
    Parameters
    ----------
    up: vector
        Primitive variables
    u : vector
        Conservative variables
    gamma : float
        specific heat ratio (default: 1.4)
    """
    rho, v, p = up
    u[0] = rho
    u[1] = rho*v
    rhoe = p / (gamma-1)
    u[2] = rhoe + 0.5*rho*v**2
    

def to_flux(u, f, gamma=1.4):
    """
    Flux calculation
    
    Parameters
    ----------
    u : vector
        Conservative variables
    f: vector
        Flux vector
    gamma : float
        specific heat ratio (default: 1.4)
    """
    rho = u[0]
    v = u[1] / u[0]
    et = u[2] / u[0]
    
    p = (gamma-1)*rho*(et - 0.5*v**2)
    ht = et + p/rho
    
    f[0] = rho*v
    f[1] = rho*v**2 + p
    f[2] = rho*v*ht
    
    return p, v

def make_flux(u, f, gamma=1.4):
    """
    Flux calculation
    
    Parameters
    ----------
    u : vector
        Conservative variables
    f: vector
        Flux vector
    gamma : float
        specific heat ratio (default: 1.4)
    """
    rho = u[0]
    v = u[1] / u[0]
    et = u[2] / u[0]
    
    p = (gamma-1)*rho*(et - 0.5*v**2)
    ht = et + p/rho
    
    f[0] = rho*v
    f[1] = rho*v**2 + p
    f[2] = rho*v*ht
    
    return f
