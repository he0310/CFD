from nbutils import array
from solvers.bc import make_bc
from solvers.rsolvers import get_rsolver
from solvers.limiter import get_limiter

import numba as nb
import numpy as np


# Very small number to prevent zero division
eps = 1e-12


def make_rhside(nfx, nfy, jac, si, sj, fluxn, bctypes, limitern='none', gamma=1.4, nvars=4, npads=(1,1)): 
    """
    Kernal generator for right hand side (First order)
    
    Parameters
    ----------
    nfx : integer
        xi 방향 격자 개수
    nfy : integer
        eta 방향 격자 개수
    jac : array
        Jacobian 크기
    si : array
        xi 방향 Face metric
    sj : array
        eta 방향 Face metric
    fluxn : string
        근사 리만 해석자 함수 이름
    bctypes : dict
        경계조건 함수 이름
    limitern : string
        Limiter 이름
    bc : function
        경계조건 함수
    nvars : integer
        벡터 크기 (기본값: 4)
    npad : integer
        경계를 위한 padding 개수 (기본값: (1,1))
    """
    bc = make_bc(nfx, nfy, si, sj, bctypes, nvars, npads)
    flux = get_rsolver(fluxn, gamma, nvars)
    phi = get_limiter(limitern)
    
    @nb.jit(nopython=True)
    def _run(u, du, ff, fg):
        # Compute BC
        bc(u)
        
        # xi sweep
        for j in range(nfy):
            jy = j + npads[1]
            for i in range(nfx+1):
                ix = i + npads[0] - 1
                
                fn = array((nvars,))
                ul, ur = array((nvars,)), array((nvars,))
                
                if (npads[0] == 1) and ((i == 0) or (i == nfx)):
                    for n in range(nvars):
                        ul[n] = u[n, jy, ix]
                        ur[n] = u[n, jy, ix+1]
                else:
                    # Implement TVD-MUSCL!!!
                    for n in range(nvars):
                        du0 = u[n, jy, ix] - u[n, jy, ix-1] + eps
                        du1 = u[n, jy, ix+1] - u[n, jy, ix] + eps
                        du2 = u[n, jy, ix+2] - u[n, jy, ix+1] + eps

                        ul[n] = u[n, jy, ix] + 0.5*phi(du0,du1)*du1
                        ur[n] = u[n, jy, ix+1] - 0.5*phi(du1,du2)*du2                       
                    
                nf = si[:, j, i]
                flux(ul, ur, nf, fn)
                ff[:, j, i] = fn
                
        # Eta sweep
        for j in range(nfy+1):
            jy = j + npads[1] - 1
            for i in range(nfx):
                ix = i + npads[0]
                
                fn = array((nvars,))
                ul, ur = array((nvars,)), array((nvars,))
                
                if (npads[1] == 1) and ((j == 0) or (j == nfy)):
                    for n in range(nvars):
                        ul[n] = u[n, jy, ix]
                        ur[n] = u[n, jy+1, ix]
                        
                else:
                    # Implement TVD-MUSCL!!!
                    for n in range(nvars):
                        du0 = u[n, jy, ix] - u[n, jy-1, ix] + eps
                        du1 = u[n, jy+1, ix] - u[n, jy, ix] + eps
                        du2 = u[n, jy+2, ix] - u[n, jy+1, ix] + eps

                        ul[n] = u[n, jy, ix] + 0.5*phi(du0,du1)*du1
                        ur[n] = u[n, jy+1, ix] - 0.5*phi(du1,du2)*du2
                    
                nf = sj[:, j, i]
                flux(ul, ur, nf, fn)
                fg[:, j, i] = fn

        # negative derivative of flux at each cell
        for j in range(nfy):
            jp = j + 1
            jy = j + npads[1]
            for i in range(nfx):        
                ip = i + 1
                ix = i + npads[0]
                
                du[:, jy, ix] = -jac[j,i]*(
                    ff[:, j, ip] - ff[:,j, i] 
                    + fg[:, jp, i] - fg[:, j, i]
                )
            
    return _run, bc
